#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Intelligent Exploration (with OpenAI, Grok, or Hugging Face Vision)
- Select via VISION_PROVIDER in .env (values: "openai", "grok", "free")

ENV vars you likely want:
  VISION_PROVIDER=free
  HF_TOKEN=hf_************************
  # Optional:
  HF_MODEL=Salesforce/blip-image-captioning-large
  HF_INFERENCE_URL=   # e.g. https://xxxxxx.us-east-1.aws.endpoints.huggingface.cloud (paid endpoints)
  HF_TIMEOUT=45

Camera/robot:
  BRUNO_CAMERA_URL=http://127.0.0.1:8080?action=stream
  BRUNO_SPEED=40
  BRUNO_TURN_SPEED=40
  BRUNO_TURN_TIME=0.5
  BRUNO_BACKUP_TIME=0.0
  ULTRA_CAUTION_CM=50
  ULTRA_DANGER_CM=25
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import base64, io, requests
from urllib.parse import urlparse
from typing import Optional, List, Dict
from datetime import datetime

# -------------------------
# Load environment variables
# -------------------------
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    print("✓ Loaded .env file")

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

# OpenAI (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
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
    "camera_url": os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream"),
    "autostart_local_stream": True,
    "stream_port": int(os.environ.get("BRUNO_STREAM_PORT", "8080")),
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm": float(os.environ.get("ULTRA_DANGER_CM", "25")),
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),
    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,
    "gpt_photo_interval": float(os.environ.get("BRUNO_PHOTO_INTERVAL", "10")),
    "gpt_vision_prompt": os.environ.get(
        "BRUNO_VISION_PROMPT",
        "Describe what you see in this image. Focus on objects, bottles, bins, obstacles, walls, and environment."
    ),
    "save_gpt_images": os.environ.get("SAVE_GPT_IMAGES", "1") not in ("0", "false", "False"),
    "save_debug": os.environ.get("SAVE_DEBUG", "1") not in ("0", "false", "False"),
    "debug_interval": int(os.environ.get("BRUNO_DEBUG_INTERVAL", "20")),
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.vision")

# =========================
# Helpers
# =========================
def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _is_stream_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def _looks_like_local(url: str, default_port: int = 8080) -> bool:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        port = u.port or (443 if u.scheme == "https" else 80)
        return u.scheme in ("http", "https") and host in ("127.0.0.1", "localhost") and port == default_port
    except Exception:
        return False

def _start_stream_background(host="0.0.0.0", port=8080):
    try:
        import live_stream_test
        t = threading.Thread(target=live_stream_test.run_stream, kwargs={"host": host, "port": port}, daemon=True)
        t.start()
        return t
    except Exception as e:
        LOG.error(f"Could not import live_stream_test: {e}")
        return None

def _wait_until(pred, timeout_s: float, interval_s: float = 0.25) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if pred(): return True
        time.sleep(interval_s)
    return False

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
        vals = []
        for _ in range(self.avg_samples):
            try:
                d_cm = self.sonar.getDistance() / 10.0
                if 2.0 <= d_cm <= 400.0: vals.append(float(d_cm))
            except Exception:
                pass
            time.sleep(self.sample_delay)

        if not vals:
            return None

        if len(vals) >= 4:
            vals.remove(max(vals))
            vals.remove(min(vals))

        if PANDAS_OK and len(vals) >= 3:
            s = pd.Series(vals); m, std = float(s.mean()), float(s.std())
            s = s[np.abs(s - m) <= std] if std > 0 else s
            if len(s) > 0:
                return float(s.mean())

        return float(np.mean(vals))

    def set_rgb(self, r, g, b):
        for i in [0, 1]:
            try:
                self.sonar.setPixelColor(i, (r, g, b))
            except Exception:
                pass

# =========================
# Vision Backends
# =========================
class VisionClient:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.provider = os.environ.get("VISION_PROVIDER", "grok").lower()
        self.last_photo_time, self.photo_count = time.time(), 0
        self.enabled = False
        self.descriptions: List[str] = []

        LOG.info(f"🔍 Initializing Vision provider = {self.provider}")
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and OPENAI_AVAILABLE:
                self.client = OpenAI()
                self.enabled = True
                LOG.info("✓ OpenAI Vision enabled")
            else:
                LOG.warning("OpenAI not available or OPENAI_API_KEY missing")
        elif self.provider == "grok":
            self.api_key = os.environ.get("XAI_API_KEY")
            if self.api_key:
                self.enabled = True
                LOG.info("✓ Grok Vision enabled")
            else:
                LOG.warning("XAI_API_KEY missing; Grok disabled")
        elif self.provider == "free":
            self.api_key = os.environ.get("HF_TOKEN")
            self.hf_endpoint = os.environ.get("HF_INFERENCE_URL", "").strip()
            self.hf_model = os.environ.get("HF_MODEL", "Salesforce/blip-image-captioning-large")
            self.hf_timeout = float(os.environ.get("HF_TIMEOUT", "45"))
            self.hf_fallback_models = [
                "Salesforce/blip-image-captioning-base",
                "nlpconnect/vit-gpt2-image-captioning"
            ]
            if self.api_key:
                self.enabled = True
                LOG.info(f"✓ Hugging Face Vision enabled (model={self.hf_model})")
            else:
                LOG.warning("HF_TOKEN not set; 'free' provider disabled")
        else:
            LOG.warning(f"Unknown provider: {self.provider}")

        # Create a folder to optionally save snapshots + captions
        if self.enabled and self.cfg.get("save_gpt_images", False):
            ensure_dir("gpt_shots")

    def should_take_photo(self) -> bool:
        return self.enabled and (time.time() - self.last_photo_time) >= self.cfg["gpt_photo_interval"]

    # ---- HF helper ----
    def _hf_caption(self, img_bytes: bytes) -> Optional[str]:
        """
        Try HF model(s) in order. Use binary upload and wait for cold starts.
        """
        def _url_for(model_id: str) -> str:
            if self.hf_endpoint:
                # If using a dedicated inference endpoint, post directly to it.
                return self.hf_endpoint.rstrip("/")
            return f"https://api-inference.huggingface.co/models/{model_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-wait-for-model": "true",
            "Accept": "application/json",
            # "Content-Type": "image/jpeg",  # optional; HF will infer for binary
        }

        candidates = [self.hf_model] + [m for m in self.hf_fallback_models if m != self.hf_model]
        for model_id in candidates:
            url = _url_for(model_id)
            try:
                r = requests.post(url, headers=headers, data=img_bytes, timeout=self.hf_timeout)
                if r.status_code in (404, 405):
                    LOG.warning(f"HF: route not found for {model_id} at {url} ({r.status_code}); trying fallback")
                    continue
                if r.status_code in (429, 500, 503):
                    time.sleep(1.5)
                    r = requests.post(url, headers=headers, data=img_bytes, timeout=self.hf_timeout)

                r.raise_for_status()
                result = r.json()
                text = None
                if isinstance(result, list) and result:
                    first = result[0]
                    if isinstance(first, dict):
                        text = first.get("generated_text") or first.get("caption")
                    elif isinstance(first, str):
                        text = first
                elif isinstance(result, dict):
                    text = result.get("generated_text") or result.get("caption") or result.get("text")

                if text and text.strip():
                    return text.strip()

                LOG.warning(f"HF: unexpected response shape for {model_id}: {result}")
            except Exception as e:
                LOG.warning(f"HF call failed for {model_id} at {url}: {e}")
        return None

    def capture_and_describe(self, frame, current_action="UNKNOWN") -> Optional[str]:
        if not self.enabled:
            return None

        # Encode to JPEG bytes
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()

        # Data-URL (for OpenAI/Grok)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{img_b64}"

        desc = None
        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":[
                        {"type":"text","text":self.cfg["gpt_vision_prompt"]},
                        {"type":"image_url","image_url":{"url":img_url,"detail":"low"}}
                    ]}],
                    max_tokens=300, temperature=0.1
                )
                desc = resp.choices[0].message.content

            elif self.provider == "grok":
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "model":"grok-2-vision-latest",
                    "messages":[{"role":"user","content":[
                        {"type":"text","text":self.cfg["gpt_vision_prompt"]},
                        {"type":"image_url","image_url":{"url":img_url}}
                    ]}],
                    "max_tokens":300,"temperature":0.1
                }
                resp = requests.post("https://api.x.ai/v1/chat/completions",
                                     headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                desc = data["choices"][0]["message"]["content"]

            elif self.provider == "free":
                desc = self._hf_caption(img_bytes)
                if not desc:
                    raise Exception("Hugging Face captioning failed across all candidates")

            else:
                desc = None

            # Record and optionally save
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.photo_count += 1
            self.last_photo_time = time.time()
            if desc:
                self.descriptions.append({"t": ts, "provider": self.provider, "text": desc})

            # Optional: persist the frame + caption
            if self.cfg.get("save_gpt_images", False):
                ensure_dir("gpt_shots")
                jpg_path = os.path.join("gpt_shots", f"{ts}_{self.provider}.jpg")
                txt_path = os.path.join("gpt_shots", f"{ts}_{self.provider}.txt")
                try:
                    with open(jpg_path, "wb") as f:
                        f.write(img_bytes)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(desc or "")
                except Exception as e:
                    LOG.warning(f"Could not save GPT shot: {e}")

            LOG.info(f"🔍 {self.provider.upper()} vision: {desc}")
            return desc

        except Exception as e:
            LOG.error(f"Vision error: {e}")
            self.last_photo_time = time.time()
            return None

# =========================
# Bruno Intelligent Explorer
# =========================
class BrunoIntelligentExplorer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK()
        self.AK.board = self.board
        self.ultra = UltrasonicRGB()
        self.vision_client = VisionClient(cfg)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_idx = 0

    def _open_camera(self) -> bool:
        if self.cfg["autostart_local_stream"] and _looks_like_local(self.cfg["camera_url"], self.cfg["stream_port"]):
            if not _is_stream_up(self.cfg["camera_url"]):
                LOG.info("Starting local stream...")
                _start_stream_background(port=self.cfg["stream_port"])
                _wait_until(lambda: _is_stream_up(self.cfg["camera_url"]), 8.0, 0.5)
        self.cap = cv2.VideoCapture(self.cfg["camera_url"])
        if not self.cap.isOpened():
            LOG.warning(f"Camera not opened: {self.cfg['camera_url']}")
            return False
        return True

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def run(self):
        LOG.info("🤖 Bruno Explorer started")
        if not self._open_camera():
            LOG.warning("No camera available")
        self.running = True
        try:
            while self.running:
                self.frame_idx += 1
                frame = None
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    frame = frame if ok else None

                d_cm = self.ultra.get_distance_cm()

                # ===== OBSTACLE AVOIDANCE & MOVEMENT =====
                if d_cm and d_cm <= self.cfg["ultra_danger_cm"]:
                    # EMERGENCY STOP
                    self.ultra.set_rgb(255, 0, 0)  # red
                    self.stop_all()
                    current_action = f"EMERGENCY STOP ({d_cm:.1f}cm)"
                    LOG.warning(current_action)
                    time.sleep(0.05)
                    continue

                elif d_cm and d_cm <= self.cfg["ultra_caution_cm"]:
                    # AVOIDANCE MANEUVER
                    self.ultra.set_rgb(255, 180, 0)  # amber

                    # Optional backup (simple strafe back-left)
                    if self.cfg["backup_time"] > 0:
                        self.car.set_velocity(self.cfg["turn_speed"], 90, 0)
                        time.sleep(self.cfg["backup_time"])
                        self.stop_all()

                    # Alternate turn direction
                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                    time.sleep(self.cfg["turn_time"])
                    self.stop_all()
                    current_action = f"ULTRA AVOID ({d_cm:.1f}cm) {'LEFT' if left else 'RIGHT'}"
                    LOG.info(current_action)
                    time.sleep(0.05)
                    continue

                else:
                    # SAFE TO MOVE FORWARD
                    self.ultra.set_rgb(0, 255, 0)  # green
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                    current_action = "FORWARD (safe)"
                    LOG.info(f"[ULTRA] {d_cm:.1f}cm - {current_action}")

                # ===== VISION SNAPSHOTS =====
                if frame is not None and self.vision_client.should_take_photo():
                    self.stop_all()
                    time.sleep(0.2)
                    LOG.info("📸 Taking GPT photo...")
                    self.vision_client.capture_and_describe(frame, current_action)
                    LOG.info("▶️ Resuming Bruno movement after photo...")
                    continue

                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
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
RUNNER = None
def _sig_handler(signum, frame):
    global RUNNER
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    RUNNER = BrunoIntelligentExplorer(CONFIG)
    RUNNER.run()
