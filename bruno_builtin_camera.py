#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Built-in Camera Surveillance (Local Captioner + LM Studio Summary)
- Based on working2.py but adapted for built-in camera
- Captions snapshots locally using captioner.py (Hugging Face pipeline)
- Collects captions for a fixed duration (default: 120s), then sends them
  to a local GPT-OSS server (LM Studio, OpenAI-compatible) for summary
- Prints the summary to console and exits cleanly
- Logs to STDOUT + ./logs/bruno.log (rotating), auto-creates ./gpt_images, ./logs, ./debug

Enhancements:
- If EMERGENCY STOP persists >5s: reverse one step (once)
- If EMERGENCY STOP persists >10s: send summary to LM Studio and exit
"""

import os, sys, time, signal, logging, subprocess, requests, threading, urllib.request
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

# ---------- Logging + paths ----------
def setup_logging_to_stdout(name: str = "bruno.builtin_camera") -> logging.Logger:
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

LOG = setup_logging_to_stdout("bruno.builtin_camera")
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
LLM_API_BASE  = os.environ.get("LLM_API_BASE", "http://192.168.1.154:1234/v1")
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

# ---------- Built-in Camera helpers ----------
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
        LOG.info(f"Starting background stream on {host}:{port}")
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

def open_builtin_camera() -> Tuple[Optional[cv2.VideoCapture], str]:
    """Open built-in camera using stream approach"""
    LOG.info("🎥 Opening built-in camera...")
    
    # Built-in camera URL
    camera_url = os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream")
    stream_port = 8080
    
    # Check if local stream is up, if not start it
    if _looks_like_local(camera_url, stream_port):
        if not _is_stream_up(camera_url):
            LOG.info(f"Local stream {camera_url} not running. Launching live_stream_test...")
            _start_stream_background(host="0.0.0.0", port=stream_port)
            ok = _wait_until(lambda: _is_stream_up(camera_url), timeout_s=8.0, interval_s=0.5)
            if not ok:
                LOG.warning("Local stream did not come up in time")
    
    # Try to open the camera stream
    methods = [
        ("Built-in stream", lambda: cv2.VideoCapture(camera_url)),
        ("Built-in stream FFMPEG", lambda: cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)),
    ]
    
    for name, func in methods:
        try:
            LOG.info(f"   Trying {name}...")
            cap = func()
            if cap and cap.isOpened():
                # Test frame read
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    LOG.info(f"   ✅ {name} SUCCESS! Resolution: {w}x{h}")
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for fresh frames
                        LOG.info("   📐 Camera properties set")
                    except Exception as e:
                        LOG.warning(f"   ⚠️  Could not set properties: {e}")
                    return cap, name
                else:
                    LOG.warning(f"   ❌ {name} opened but cannot read frames")
                    cap.release()
            else:
                LOG.warning(f"   ❌ {name} failed to open")
                if cap:
                    cap.release()
        except Exception as e:
            LOG.warning(f"   ❌ {name} error: {e}")
    
    LOG.error("❌ Failed to open built-in camera")
    return None, "Failed"

def find_working_builtin_camera() -> Tuple[Optional[cv2.VideoCapture], str]:
    """Find and open the built-in camera"""
    LOG.info("🔍 Scanning for built-in camera...")
    
    # Try multiple attempts for built-in camera
    for attempt in range(3):
        LOG.info(f"📹 Built-in camera connection attempt {attempt + 1}...")
        cap, method = open_builtin_camera()
        if cap:
            LOG.info(f"✅ Successfully opened built-in camera using {method}")
            return cap, method
        if attempt < 2:
            LOG.info("⏳ Retrying in 2 seconds...")
            time.sleep(2.0)
    
    LOG.error("❌ No working built-in camera found")
    return None, "Failed"

# ---------- Config ----------
CONFIG = {
    "builtin_camera_only": True,
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
    # Try different LM Studio endpoints
    base_url = LLM_API_BASE.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]  # Remove /v1 if present
    
    possible_urls = [
        f"{base_url}/v1/chat/completions",  # Standard OpenAI compatible
        f"{base_url}/chat/completions",     # Without v1
        f"{base_url}/api/chat/completions", # Alternative path
        f"{base_url}/v1/completions",       # Legacy completions endpoint
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'lm-studio')}"
    }
    
    # Try each endpoint until one works
    last_error = None
    for url in possible_urls:
        try:
            LOG.info(f"Trying LM Studio endpoint: {url}")
            r = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            
            # Debug the response structure
            LOG.info(f"LM Studio response keys: {list(data.keys())}")
            
            if "choices" in data and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    summary = data["choices"][0]["message"]["content"]
                    LOG.info(f"✅ LM Studio summary successful via {url}")
                    return summary.strip()
                elif "text" in data["choices"][0]:
                    summary = data["choices"][0]["text"]
                    LOG.info(f"✅ LM Studio summary successful via {url}")
                    return summary.strip()
            
            # Fallback: try to find content anywhere in response
            if "content" in data:
                LOG.info(f"✅ LM Studio summary successful via {url} (fallback)")
                return data["content"].strip()
            
            # If we get here, response format was unexpected but not an error
            LOG.warning(f"Unexpected response format from {url}: {data}")
            last_error = f"Unexpected response format: {data}"
            
        except requests.exceptions.RequestException as e:
            LOG.warning(f"Connection failed for {url}: {e}")
            last_error = f"Connection failed: {e}"
            continue
        except Exception as e:
            LOG.warning(f"Error with {url}: {e}")
            last_error = f"Error: {e}"
            continue
    
    return f"[LM Studio summary failed] All endpoints failed. Last error: {last_error}"

# ---------- Main ----------
class BrunoBuiltinCameraSurveillance:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        self.ultra = UltrasonicRGB()
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_method: str = "Unknown"
        self.running = False
        self.frame_idx = 0

        self.snapshotter = Snapshotter(cfg["photo_interval"])
        self.start_time = time.time()
        self.summary_due_at = self.start_time + SUMMARY_DELAY
        self.captions: List[Dict] = []

        # Emergency tracking
        self.emergency_start_time: Optional[float] = None
        self.emergency_reversed_once: bool = False

    def _open_camera(self) -> bool:
        LOG.info("🎥 Opening built-in camera...")
        for attempt in range(self.cfg["camera_retry_attempts"]):
            LOG.info(f"📹 Built-in camera connection attempt {attempt + 1}...")
            self.cap, self.camera_method = find_working_builtin_camera()
            if self.cap:
                LOG.info(f"✅ Built-in camera connected using: {self.camera_method}")
                return True
            if attempt < self.cfg["camera_retry_attempts"] - 1:
                LOG.info(f"⏳ Retrying in {self.cfg['camera_retry_delay']} seconds...")
                time.sleep(self.cfg["camera_retry_delay"])
        LOG.error("❌ Failed to connect built-in camera")
        return False

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def _do_snapshot_and_caption(self, frame) -> None:
        # Add frame hash to detect if we're reusing the same frame
        frame_hash = hash(frame.tobytes())
        LOG.info(f"📸 Processing frame with hash: {frame_hash}")
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_path = save_image_path("builtin_camera_photo")
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

        self.captions.append({"timestamp": ts, "path": str(img_path), "caption": cap_text, "frame_hash": frame_hash})

    def _finish_with_summary_and_exit(self, reason: str):
        LOG.warning(f"⏹️  Triggering early summary due to: {reason}")
        summary = summarize_captions_lmstudio(self.captions)
        print("\n" + "=" * 80)
        print("🧾 SUMMARY (LM Studio):")
        print(summary)
        print("=" * 80 + "\n")
        self.shutdown()
        sys.exit(0)

    def _maybe_finish_with_summary(self):
        if time.time() >= self.summary_due_at:
            self._finish_with_summary_and_exit("time window reached")

    def run(self):
        LOG.info("🤖 Bruno Built-in Camera Surveillance (local captioner → LM Studio summary)")
        LOG.info(f"🗂  Images will be saved to: {paths['gpt_images']}")
        LOG.info(f"🗂  Logs will be saved to: {paths['logs']}/bruno.log")
        LOG.info(f"🧠 Caption model: {os.environ.get('CAPTION_MODEL', 'Salesforce/blip-image-captioning-large')}")
        LOG.info(f"🕒 Snapshot interval: {self.cfg['photo_interval']}s | Summary at: {SUMMARY_DELAY}s")

        if not self._open_camera():
            LOG.error("❌ Cannot start without built-in camera")
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

                        # Get a completely fresh frame from built-in camera stream
                        fresh = None
                        if self.cap and self.cap.isOpened():
                            # Aggressive buffer flushing for stream-based cameras
                            LOG.info("🔄 Flushing camera buffer for fresh frame...")
                            for flush_attempt in range(5):
                                try:
                                    ret, _ = self.cap.read()  # Discard old buffered frames
                                    if not ret:
                                        break
                                    time.sleep(0.05)  # Allow buffer to update
                                except Exception:
                                    break
                            
                            # Now get the fresh frame
                            for fresh_attempt in range(3):
                                try:
                                    ret, fresh = self.cap.read()
                                    if ret and fresh is not None:
                                        LOG.info(f"✅ Got fresh frame from built-in camera (attempt {fresh_attempt + 1})")
                                        break
                                    time.sleep(0.1)
                                except Exception as e:
                                    LOG.warning(f"Fresh frame attempt {fresh_attempt + 1} failed: {e}")
                        
                        if fresh is not None:
                            use_frame = fresh
                            LOG.info("📸 Using fresh captured frame")
                        else:
                            use_frame = last_good_frame
                            LOG.warning("⚠️ Using last good frame (could not get fresh frame)")
                        self._do_snapshot_and_caption(use_frame)

                        # Resume forward if ultrasonic is currently safe (or unknown)
                        if last_distance_cm is None or last_distance_cm > self.cfg["ultra_caution_cm"]:
                            self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                        time.sleep(0.2)
                    else:
                        LOG.warning("⚠️  Snapshot due, but no frame available yet (skipping).")

                # 3) Ultrasonic avoidance + EMERGENCY persistence logic
                d_cm = self.ultra.get_distance_cm()
                last_distance_cm = d_cm

                if d_cm is not None and d_cm <= self.cfg["ultra_danger_cm"]:
                    # Entering or staying in EMERGENCY
                    if self.emergency_start_time is None:
                        self.emergency_start_time = time.time()
                        self.emergency_reversed_once = False
                        LOG.warning(f"EMERGENCY STOP ({d_cm:.1f} cm) — timer started")
                    else:
                        LOG.warning(f"EMERGENCY STOP ({d_cm:.1f} cm) — ongoing")

                    self.ultra.set_rgb(255, 0, 0)
                    self.stop_all()
                    time.sleep(0.02)

                    elapsed = time.time() - self.emergency_start_time

                    # After 5s stuck: reverse one step (only once)
                    if (elapsed >= 5.0) and (not self.emergency_reversed_once):
                        LOG.warning("⏪ EMERGENCY >5s — reversing one step")
                        try:
                            # Reverse burst (~0.6s). If negative speed isn't supported, reduce speed or tweak duration.
                            self.car.set_velocity(-self.cfg["forward_speed"], 90, 0)
                            time.sleep(0.6)
                        except Exception:
                            pass
                        finally:
                            self.stop_all()
                            self.emergency_reversed_once = True

                    # After 30s stuck: send summary and exit
                    if elapsed >= 30.0:
                        self._finish_with_summary_and_exit("prolonged emergency stop (>30s)")

                    # Skip the rest of loop; still check time-based summary below
                elif d_cm is not None and d_cm <= self.cfg["ultra_caution_cm"]:
                    # Caution: avoidance
                    self.ultra.set_rgb(255, 180, 0)

                    # Reset emergency timer since not in danger anymore
                    if self.emergency_start_time is not None:
                        LOG.info("✅ Left EMERGENCY zone — timer reset")
                    self.emergency_start_time = None
                    self.emergency_reversed_once = False

                    if self.cfg["backup_time"] > 0:
                        self.car.set_velocity(self.cfg["turn_speed"], 90, 0)
                        time.sleep(self.cfg["backup_time"])
                        self.stop_all()

                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                    time.sleep(self.cfg["turn_time"])
                    self.stop_all()
                    LOG.info(f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}")
                else:
                    # Safe: drive forward
                    self.ultra.set_rgb(0, 255, 0)

                    # Reset emergency state if we were in it
                    if self.emergency_start_time is not None:
                        LOG.info("✅ Safe distance — emergency cleared")
                    self.emergency_start_time = None
                    self.emergency_reversed_once = False

                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # 4) (Optional) Vision avoidance could go here

                # 5) Maybe finish with summary at the scheduled window
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
RUNNER: Optional[BrunoBuiltinCameraSurveillance] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping...")
    if RUNNER:
        RUNNER.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno Built-in Camera Surveillance (local captioner → LM Studio summary)")
    LOG.info("Press Ctrl+C to stop")
    RUNNER = BrunoBuiltinCameraSurveillance(CONFIG)
    RUNNER.run()