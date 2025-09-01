#!/usr/bin/env python3
# coding: utf-8
"""
Bruno External Camera Surveillance (Snapshots Always, Preemptive Photos, STDOUT Logging)
- Logs to STDOUT + ./logs/bruno.log (rotating)
- Auto-creates ./gpt_images, ./logs, ./debug relative to CWD
- External USB camera robust open
- Pre-emptive snapshots (taken before avoidance branches)
- Saves snapshots even if GPT Vision is disabled; analyzes when enabled
"""

import os, sys, time, json, signal, logging, threading, urllib.request, subprocess, base64, io
from urllib.parse import urlparse
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from logging.handlers import RotatingFileHandler

# =========================
# Logging to STDOUT + ensure relative folders
# =========================
def setup_logging_to_stdout(name: str = "bruno.external_camera") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Avoid duplicates on reload
    if logger.handlers:
        logger.handlers.clear()

    # Stream to STDOUT
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    # File (./logs/bruno.log) with rotation
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(logs_dir / "bruno.log"), maxBytes=5_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("Logging initialized (stdout + ./logs/bruno.log)")
    return logger

def ensure_relative_paths() -> Dict[str, Path]:
    """Create relative output dirs next to where the process is started (CWD)."""
    base = Path.cwd()
    paths = {
        "base": base,
        "gpt_images": base / "gpt_images",
        "logs": base / "logs",
        "debug": base / "debug",
    }
    for p in paths.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return paths

LOG = setup_logging_to_stdout("bruno.external_camera")
paths = ensure_relative_paths()
LOG.info(f"Working directory: {paths['base']}")
LOG.info(f"Images → {paths['gpt_images']}")
LOG.info(f"Logs   → {paths['logs']}")

def save_image_path(prefix: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    us = int((time.time() * 1_000_000) % 1_000_000)
    return paths["gpt_images"] / f"{prefix}_{ts}_{us:06d}.jpg"

# =========================
# Environment
# =========================
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    LOG.info("✓ Loaded .env file")

# --- Hiwonder SDK path ---
sys.path.append('/home/pi/MasterPi')

# =========================
# Dependencies
# =========================
import cv2
import numpy as np
from PIL import Image

# Optional pandas smoothing
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
    LOG.info("✓ OpenAI library available")
except ImportError:
    LOG.warning("✗ OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Hiwonder modules
import common.sonar as Sonar
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board
from kinematics.transform import *
from kinematics.arm_move_ik import ArmIK

# =========================
# External Camera Helpers
# =========================
def get_available_cameras() -> List[Dict]:
    cams = []
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            cams.append({
                'path': device_path,
                'index': i,
                'accessible': os.access(device_path, os.R_OK)
            })
    return cams

def fix_camera_permissions():
    LOG.info("🔧 Checking camera permissions...")
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if result.returncode == 0 and 'video' not in result.stdout:
            LOG.warning("⚠️  User not in 'video' group. Run: sudo usermod -a -G video $USER && re-login")
        for i in range(5):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path) and not os.access(device_path, os.R_OK):
                try:
                    subprocess.run(['sudo', 'chmod', '666', device_path], timeout=5)
                    LOG.info(f"✅ Fixed permissions for {device_path}")
                except Exception as e:
                    LOG.warning(f"⚠️  Could not fix permissions for {device_path}: {e}")
    except Exception as e:
        LOG.warning(f"⚠️  Permission check failed: {e}")

def open_external_camera_robust(device_info: Dict) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    device_path = device_info['path']; device_index = device_info['index']
    LOG.info(f"🎥 Attempting to open {device_path} (index {device_index})...")

    methods = [
        ('V4L2 with index', lambda: cv2.VideoCapture(device_index, cv2.CAP_V4L2)),
        ('V4L2 with path',  lambda: cv2.VideoCapture(device_path, cv2.CAP_V4L2)),
        ('Default with index', lambda: cv2.VideoCapture(device_index)),
        ('Default with path',  lambda: cv2.VideoCapture(device_path)),
        ('GStreamer pipeline',
         lambda: cv2.VideoCapture(f'v4l2src device={device_path} ! videoconvert ! appsink', cv2.CAP_GSTREAMER)),
    ]

    for name, func in methods:
        try:
            LOG.info(f"   Trying {name}...")
            cap = func()
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    LOG.info(f"   ✅ {name} SUCCESS! Resolution: {w}x{h}")
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        LOG.info("   📐 Camera properties set")
                    except Exception as e:
                        LOG.warning(f"   ⚠️  Could not set properties: {e}")
                    return cap, name
                else:
                    LOG.warning(f"   ❌ {name} opened but cannot read frames")
                    cap.release()
            else:
                LOG.warning(f"   ❌ {name} failed to open")
                if cap: cap.release()
        except Exception as e:
            LOG.warning(f"   ❌ {name} error: {e}")
    return None, None

def find_working_external_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[Dict]]:
    LOG.info("🔍 Scanning for external cameras...")
    fix_camera_permissions()
    cams = get_available_cameras()
    if not cams:
        LOG.error("❌ No camera devices found at /dev/video*")
        return None, None

    LOG.info(f"📹 Found {len(cams)} camera device(s)")
    for camera in cams:
        if not camera['accessible']:
            LOG.warning(f"⚠️  {camera['path']} not accessible (permissions issue)")
            continue
        cap, method = open_external_camera_robust(camera)
        if cap:
            LOG.info(f"✅ Successfully opened {camera['path']} using {method}")
            return cap, camera

    LOG.error("❌ No working external cameras found")
    return None, None

# =========================
# Config
# =========================
CONFIG = {
    "external_camera_only": True,
    "camera_retry_attempts": 3,
    "camera_retry_delay": 2.0,

    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM", "25")),
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),

    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,

    "gpt_photo_interval": 15,
    "gpt_vision_prompt": "Describe what you see in this image. Focus on: objects, furniture, rooms, people, pets, bottles, bins/containers, obstacles, walls, and the general environment. Be concise but detailed.",
    "save_gpt_images": True,

    "save_debug": True,
    "debug_interval": 20,
}

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
# Vision obstacle detection
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
# GPT Vision (snapshots always; analysis only if enabled)
# =========================
class GPTVision:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.enabled = False
        self.client = None

        LOG.info("🔍 Initializing GPT Vision...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.enabled = True
                LOG.info("✅ GPT Vision ENABLED")
                LOG.info(f"📸 Photo interval: {cfg['gpt_photo_interval']} seconds")
            except Exception as e:
                LOG.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.enabled = False
        else:
            LOG.info("🔕 GPT Vision disabled — missing OPENAI_API_KEY or openai lib")

        self.last_photo_time = 0
        self.photo_count = 0

    def should_take_photo(self) -> bool:
        """Schedule snapshots by time even if GPT is disabled."""
        now = time.time()
        if self.photo_count == 0:
            return True
        return (now - self.last_photo_time) >= self.cfg["gpt_photo_interval"]

    def capture_and_describe(self, frame: np.ndarray, current_action: str = "UNKNOWN") -> Optional[str]:
        """Always save a snapshot; only call OpenAI when enabled."""
        try:
            now = time.time()
            self.photo_count += 1

            # 1) Save snapshot unconditionally
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            filename = save_image_path("external_camera_photo")
            image.save(str(filename))
            LOG.info(f"💾 Snapshot saved: {filename}")

            # 2) If GPT is disabled, just update timer and return
            if not self.enabled:
                self.last_photo_time = now
                LOG.info("🔕 GPT disabled — photo saved without analysis")
                return None

            # 3) If GPT is enabled, send to API
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_data = f"data:image/jpeg;base64,{base64_data}"

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.cfg["gpt_vision_prompt"]},
                        {"type": "image_url", "image_url": {"url": image_data, "detail": "low"}},
                    ],
                }],
                max_tokens=300,
                temperature=0.1,
            )
            description = resp.choices[0].message.content
            self.last_photo_time = now
            LOG.info(f"🔍 GPT Analysis #{self.photo_count}: {description}")
            return description

        except Exception as e:
            LOG.error(f"❌ GPT Vision error: {e}")
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
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_info: Optional[Dict] = None
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
        LOG.info("🤖 Bruno External Camera Surveillance (snapshots always, preemptive photos)")
        LOG.info("📹 Optimized for external USB cameras")
        LOG.info(f"🗂  Images will be saved to: {paths['gpt_images']}")
        LOG.info(f"🗂  Logs will be saved to: {paths['logs']}/bruno.log")

        if not self._open_camera():
            LOG.error("❌ Cannot start without external camera")
            return

        self.running = True
        last_good_frame = None

        try:
            while self.running:
                self.frame_idx += 1
                frame = None

                # 1) Grab a frame ASAP
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        LOG.warning("📹 Camera read failed; attempting reconnection...")
                        self.cap.release()
                        time.sleep(1)
                        self._open_camera()
                    else:
                        last_good_frame = frame

                # 2) PHOTO FIRST — pre-emptive capture even if we will soon avoid/stop
                if self.gpt_vision.should_take_photo():
                    if last_good_frame is not None:
                        LOG.info("🛑 STOP for external camera photo session (pre-emptive)…")
                        self.stop_all()
                        time.sleep(0.2)

                        # Flush camera buffer and get a fresh frame if possible
                        fresh = None
                        if self.cap and self.cap.isOpened():
                            for _ in range(3):
                                self.cap.read(); time.sleep(0.03)
                            for attempt in range(3):
                                ok, fresh = self.cap.read()
                                if ok and fresh is not None:
                                    LOG.info(f"✅ Fresh frame for photo on attempt {attempt + 1}")
                                    break
                                time.sleep(0.05)

                        use_frame = fresh if fresh is not None else last_good_frame
                        desc = self.gpt_vision.capture_and_describe(use_frame, "PREEMPTIVE_SNAPSHOT")
                        if desc:
                            LOG.info("✅ Photo processing OK")
                        else:
                            LOG.info("✅ Snapshot saved (no analysis or analysis failed)")
                        time.sleep(0.3)
                    else:
                        LOG.warning("⚠️  Photo due, but no frame available yet (skipping this cycle)")

                # 3) ULTRASONIC AVOIDANCE
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

                # 4) VISION AVOIDANCE
                current_action = "FORWARD (ultra safe)"
                if last_good_frame is not None and self.cfg["use_vision"]:
                    obs = _vision_obstacles(last_good_frame, self.cfg)
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
            if self.cap: self.cap.release()
        except Exception:
            pass
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass

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
    LOG.info("🤖 Bruno External Camera Surveillance (snapshots always, preemptive photos)")
    LOG.info("Press Ctrl+C to stop")
    RUNNER = BrunoExternalCameraSurveillance(CONFIG)
    RUNNER.run()
