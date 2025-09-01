#!/usr/bin/env python3
# coding: utf-8
"""
Bruno External Camera Surveillance (Local Captioner + LM Studio Summary)
- Captions snapshots locally using captioner.py (Hugging Face pipeline)
- Collects captions for a fixed duration (default: 120s), then sends them
  to a local GPT-OSS server (LM Studio, OpenAI-compatible) for summary
- Prints the summary to console and exits cleanly
- Logs to STDOUT + ./logs/bruno.log (rotating), auto-creates ./gpt_images, ./logs, ./debug
"""

import os, sys, time, signal, logging, subprocess, requests
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ---------- Logging + paths ----------
def setup_logging_to_stdout(name: str = "bruno.external_camera") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(str(logs_dir / "bruno.log"), maxBytes=5_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info("Logging initialized (stdout + ./logs/bruno.log)")
    return logger

def ensure_relative_paths() -> Dict[str, Path]:
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

def sidecar_txt_path(img_path: Path) -> Path:
    return img_path.with_suffix(".txt")

# ---------- Environment + deps ----------
# Load .env if present
if os.path.exists(".env"):
    for line in Path(".env").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k] = v
    LOG.info("✓ Loaded .env file")

PHOTO_INTERVAL = int(os.environ.get("PHOTO_INTERVAL_SEC", "15"))
SUMMARY_DELAY = int(os.environ.get("SUMMARY_DELAY_SEC", "120"))  # 2 minutes by default
LLM_API_BASE  = os.environ.get("LLM_API_BASE", "http://localhost:1234/v1")
LLM_MODEL     = os.environ.get("LLM_MODEL", "lmstudio")
LLM_TIMEOUT   = float(os.environ.get("LLM_TIMEOUT_SEC", "30"))

# Hiwonder SDK path (movement/ultrasonic)
sys.path.append("/home/pi/MasterPi")

import cv2
import numpy as np
from PIL import Image

try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    pd = None
    PANDAS_OK = False

# Local captioner (Hugging Face BLIP or fallback as provided)
import captioner  # your captioner.py in same directory

# Hiwonder modules
import common.sonar as Sonar
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board
from kinematics.transform import *
from kinematics.arm_move_ik import ArmIK

# ---------- Camera helpers ----------
def get_available_cameras() -> List[Dict]:
    cams = []
    for i in range(10):
        dp = f"/dev/video{i}"
        if os.path.exists(dp):
            cams.append({"path": dp, "index": i, "accessible": os.access(dp, os.R_OK)})
    return cams

def fix_camera_permissions():
    LOG.info("🔧 Checking camera permissions...")
    try:
        result = subprocess.run(["groups"], capture_output=True, text=True)
        if result.returncode == 0 and "video" not in result.stdout:
            LOG.warning("⚠️  User not in 'video' group. Run: sudo usermod -a -G video $USER && re-login")
        for i in range(5):
            dp = f"/dev/video{i}"
            if os.path.exists(dp) and not os.access(dp, os.R_OK):
                try:
                    subprocess.run(["sudo", "chmod", "666", dp], timeout=5)
                    LOG.info(f"✅ Fixed permissions for {dp}")
                except Exception as e:
                    LOG.warning(f"⚠️  Could not fix permissions for {dp}: {e}")
    except Exception as e:
        LOG.warning(f"⚠️  Permission check failed: {e}")

def open_external_camera_robust(device_info: Dict) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    dp = device_info["path"]; idx = device_info["index"]
    LOG.info(f"🎥 Attempting to open {dp} (index {idx})...")

    methods = [
        ("V4L2 with index",  lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ("V4L2 with path",   lambda: cv2.VideoCapture(dp,  cv2.CAP_V4L2)),
        ("Default with index", lambda: cv2.VideoCapture(idx)),
        ("Default with path",  lambda: cv2.VideoCapture(dp)),
        ("GStreamer pipeline",
         lambda: cv2.VideoCapture(f"v4l2src device={dp} ! videoconvert ! appsink", cv2.CAP_GSTREAMER)),
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
    for c in cams:
        if not c["accessible"]:
            LOG.warning(f"⚠️  {c['path']} not accessible (permissions issue)")
            continue
    for c in sorted(cams, key=lambda x: x["index"]):
        cap, method = open_external_camera_robust(c)
        if cap:
            LOG.info(f"✅ Successfully opened {c['path']} using {method}")
            return cap, c
    LOG.error("❌ No working external cameras found")
    return None, None

# ---------- Config ----------
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

    "photo_interval": PHOTO_INTERVAL,
    "save_debug": True,
    "debug_interval": 20,
}

# ---------- Ultrasonic ----------
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
            s = pd.Series(vals); m, std = float(s.mean()), float(s.std())
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

# ---------- Vision obstacles ----------
def _estimate_px_distance(w: int, h: int, y: int, H: int) -> float:
    size = (w * h) / (100 * 100)
    pos  = (H - y) / max(H, 1)
    prox = size * 0.6 + pos * 0.4
    return max(20, 200 - prox * 100)

def _vision_obstacles(frame, cfg: Dict) -> List[Dict]:
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

# ---------- Snapshot scheduler ----------
class Snapshotter:
    def __init__(self, interval_s: int):
        self.interval_s = interval_s
        self.last_t = 0.0
        self.count = 0
    def due(self) -> bool:
        now = time.time()
        if self.count == 0:
            return True
        return (now - self.last_t) >= self.interval_s
    def mark(self):
        self.last_t = time.time()
        self.count += 1

# ---------- LM Studio (OpenAI-compatible) summary ----------
def summarize_captions_lmstudio(captions: List[Dict]) -> str:
    """
    Send captions to a local OpenAI-compatible server (LM Studio) and return a summary.
    Each caption dict: {'timestamp': 'HH:MM:SS', 'path': '/abs/...jpg', 'caption': '...'}
    """
    if not captions:
        return "No captions were captured in the time window."

    bullet_lines = [f"- [{c['timestamp']}] {c['caption']}" for c in captions]
    user_text = (
        "You are a concise surveillance summarizer.\n"
        "Given the following image captions collected over ~2 minutes, "
        "summarize the scene: key objects, activities, potential hazards, and overall context in 3–6 bullet points.\n\n"
        "CAPTIONS:\n" + "\n".join(bullet_lines)
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You write crisp, factual summaries for surveillance operators."},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    url = LLM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'lm-studio')}"
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        summary = data["choices"][0]["message"]["content"]
        return summary.strip()
    except Exception as e:
        return f"[LM Studio summary failed] {e}"

# ---------- Main ----------
class BrunoExternalCameraSurveillance:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        self.ultra = UltrasonicRGB()
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_info: Optional[Dict] = None
        self.running = False
        self.frame_idx = 0

        self.snapshotter = Snapshotter(cfg["photo_interval"])
        self.start_time = time.time()
        self.summary_due_at = self.start_time + SUMMARY_DELAY
        self.captions: List[Dict] = []

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

    def _do_snapshot_and_caption(self, frame) -> None:
        """Save a snapshot and caption it with captioner.get_caption; store result."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_path = save_image_path("external_camera_photo")
        image.save(str(img_path))
        self.snapshotter.mark()
        LOG.info(f"💾 Snapshot saved: {img_path}")

        cap_text = captioner.get_caption(str(img_path))
        LOG.info(f"📝 Caption: {cap_text}")

        ts = time.strftime("%H:%M:%S")
        txt_path = sidecar_txt_path(img_path)
        try:
            txt_path.write_text(cap_text + "\n", encoding="utf-8")
            LOG.info(f"💾 Caption saved: {txt_path}")
        except Exception as e:
            LOG.warning(f"⚠️  Failed to write caption file: {e}")

        self.captions.append({"timestamp": ts, "path": str(img_path), "caption": cap_text})

    def _maybe_finish_with_summary(self):
        """If 2 minutes have elapsed, send captions to LM Studio, print summary, and exit."""
        if time.time() >= self.summary_due_at:
            LOG.info("⏱️  Summary window reached — generating summary via LM Studio…")
            summary = summarize_captions_lmstudio(self.captions)
            print("\n" + "=" * 80)
            print("🧾 SUMMARY (LM Studio):")
            print(summary)
            print("=" * 80 + "\n")

            # Clean exit
            self.shutdown()
            sys.exit(0)

    def run(self):
        LOG.info("🤖 Bruno External Camera Surveillance (local captioner → LM Studio summary)")
        LOG.info(f"🗂  Images will be saved to: {paths['gpt_images']}")
        LOG.info(f"🗂  Logs will be saved to: {paths['logs']}/bruno.log")
        LOG.info(f"🧠 Caption model: {os.environ.get('CAPTION_MODEL', 'Salesforce/blip-image-captioning-large')}")
        LOG.info(f"🕒 Snapshot interval: {self.cfg['photo_interval']}s | Summary at: {SUMMARY_DELAY}s")

        if not self._open_camera():
            LOG.error("❌ Cannot start without external camera")
            return

        self.running = True
        last_good_frame = None
        last_distance_cm: Optional[float] = None  # track ultrasonic to know when to resume

        try:
            while self.running:
                self.frame_idx += 1
                frame = None

                # 1) Grab frame
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        LOG.warning("📹 Camera read failed; attempting reconnection...")
                        self.cap.release()
                        time.sleep(1)
                        self._open_camera()
                    else:
                        last_good_frame = frame

                # 2) Pre-emptive snapshot+caption (before avoidance)
                if self.snapshotter.due():
                    if last_good_frame is not None:
                        LOG.info("🛑 STOP for snapshot (pre-emptive)…")
                        self.stop_all()
                        time.sleep(0.15)

                        # Try to get a fresh frame, else fallback
                        fresh = None
                        if self.cap and self.cap.isOpened():
                            for _ in range(3):
                                self.cap.read(); time.sleep(0.02)
                            for _ in range(3):
                                ok, fresh = self.cap.read()
                                if ok and fresh is not None:
                                    break
                                time.sleep(0.03)
                        use_frame = fresh if fresh is not None else last_good_frame
                        self._do_snapshot_and_caption(use_frame)

                        # ✅ Resume forward if ultrasonic is currently safe (or unknown)
                        if last_distance_cm is None or last_distance_cm > self.cfg["ultra_caution_cm"]:
                            self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                        time.sleep(0.2)
                    else:
                        LOG.warning("⚠️  Snapshot due, but no frame available yet (skipping).")

                # 3) Ultrasonic avoidance
                d_cm = self.ultra.get_distance_cm()
                last_distance_cm = d_cm  # <-- track latest distance for resume logic

                if d_cm is not None and d_cm <= self.cfg["ultra_danger_cm"]:
                    self.ultra.set_rgb(255, 0, 0)
                    self.stop_all()
                    LOG.warning(f"EMERGENCY STOP ({d_cm:.1f} cm)")
                    time.sleep(0.05)
                    # continue; summary check still happens below

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
                    LOG.info(f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}")
                    # continue; summary check still happens below
                else:
                    # ✅ SAFE → drive forward
                    self.ultra.set_rgb(0, 255, 0)
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # 4) (Optional) Vision avoidance — keep simple or extend as needed
                # If you want to re-enable full vision avoidance, insert it here and
                # only override motion when an obstacle is detected.

                # 5) Maybe finish with summary at the 2-minute mark
                self._maybe_finish_with_summary()

                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
            self.shutdown()
        except SystemExit:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error: {e}")
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

# ---------- Entrypoint ----------
RUNNER: Optional[BrunoExternalCameraSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping...")
    if RUNNER:
        RUNNER.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno External Camera Surveillance (local captioner → LM Studio summary)")
    LOG.info("Press Ctrl+C to stop")
    RUNNER = BrunoExternalCameraSurveillance(CONFIG)
    RUNNER.run()
