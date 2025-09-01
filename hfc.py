#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Intelligent Exploration (with OpenAI, Grok, or Hugging Face BLIP Vision)
- Select via VISION_PROVIDER in .env (values: "openai", "grok", "free")
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import base64, io, requests
from urllib.parse import urlparse
from typing import Optional, List, Dict

# Load environment variables
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

# OpenAI
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
    "stream_port": 8080,
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
    "gpt_photo_interval": 10,
    "gpt_vision_prompt": "Describe what you see in this image. Focus on objects, bottles, bins, obstacles, walls, and environment.",
    "save_gpt_images": True,
    "save_debug": True,
    "debug_interval": 20,
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.vision")

# =========================
# Helpers
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
            except Exception: pass
            time.sleep(self.sample_delay)
        if not vals: return None
        if len(vals) >= 4: vals.remove(max(vals)); vals.remove(min(vals))
        if PANDAS_OK and len(vals) >= 3:
            s = pd.Series(vals); m, std = float(s.mean()), float(s.std())
            s = s[np.abs(s - m) <= std] if std > 0 else s
            if len(s) > 0: return float(s.mean())
        return float(np.mean(vals))

    def set_rgb(self, r, g, b):
        for i in [0, 1]:
            try: self.sonar.setPixelColor(i, (r, g, b))
            except Exception: pass

# =========================
# Vision Backends
# =========================
class VisionClient:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.provider = os.environ.get("VISION_PROVIDER", "grok").lower()
        self.last_photo_time, self.photo_count = time.time(), 0
        self.enabled = False
        self.descriptions = []

        LOG.info(f"🔍 Initializing Vision provider = {self.provider}")
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and OPENAI_AVAILABLE:
                self.client = OpenAI()
                self.enabled = True
                LOG.info("✓ OpenAI Vision enabled")
        elif self.provider == "grok":
            self.api_key = os.environ.get("XAI_API_KEY")
            if self.api_key:
                self.enabled = True
                LOG.info("✓ Grok Vision enabled")
        elif self.provider == "free":
            self.api_key = os.environ.get("HF_TOKEN")
            if self.api_key:
                self.enabled = True
                LOG.info("✓ Hugging Face BLIP Vision enabled")
        else:
            LOG.warning(f"Unknown provider: {self.provider}")

    def should_take_photo(self) -> bool:
        return self.enabled and (time.time() - self.last_photo_time) >= self.cfg["gpt_photo_interval"]

    def capture_and_describe(self, frame, current_action="UNKNOWN"):
        if not self.enabled: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb); buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{img_b64}"

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
                resp.raise_for_status(); data = resp.json()
                desc = data["choices"][0]["message"]["content"]

            elif self.provider == "free":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Use the specific model you requested
                api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
                
                # Convert image bytes to base64 for JSON payload
                b64_image = base64.b64encode(img_bytes).decode('utf-8')
                
                # Try different payload formats
                payload_formats = [
                    # Format 1: Direct image data
                    {"inputs": b64_image},
                    # Format 2: Image URL format (if we had URL)
                    # {"inputs": {"image": b64_image}},
                    # Format 3: Parameters format
                    {"inputs": b64_image, "parameters": {"max_length": 50}}
                ]
                
                desc = None
                for i, payload in enumerate(payload_formats):
                    try:
                        LOG.info(f"Trying HF API format {i+1}...")
                        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
                        resp.raise_for_status()
                        result = resp.json()
                        
                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                                desc = result[0]['generated_text']
                            elif isinstance(result[0], str):
                                desc = result[0]
                        elif isinstance(result, dict):
                            desc = result.get('generated_text', '') or result.get('text', '') or str(result)
                        
                        if desc and desc.strip():
                            LOG.info(f"✓ HF API success with format {i+1}")
                            break
                            
                    except Exception as e:
                        LOG.warning(f"HF API format {i+1} failed: {e}")
                        if i == 0:  # For first failure, also try binary data
                            try:
                                LOG.info("Trying binary data format...")
                                headers_binary = {"Authorization": f"Bearer {self.api_key}"}
                                resp = requests.post(api_url, headers=headers_binary, data=img_bytes, timeout=30)
                                resp.raise_for_status()
                                result = resp.json()
                                
                                if isinstance(result, list) and len(result) > 0:
                                    desc = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
                                elif isinstance(result, dict):
                                    desc = result.get('generated_text', '') or str(result)
                                
                                if desc and desc.strip():
                                    LOG.info("✓ HF API success with binary format")
                                    break
                                    
                            except Exception as e2:
                                LOG.warning(f"Binary format also failed: {e2}")
                        continue
                
                if not desc or not desc.strip():
                    raise Exception("Hugging Face BLIP API failed with all formats")

            else:
                desc = None

            self.photo_count += 1; self.last_photo_time = time.time()
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
        self.cfg, self.car, self.board = cfg, mecanum.MecanumChassis(), Board()
        self.AK = ArmIK(); self.AK.board = self.board
        self.ultra, self.vision_client = UltrasonicRGB(), VisionClient(cfg)
        self.cap, self.running, self.frame_idx = None, False, 0

    def _open_camera(self) -> bool:
        if self.cfg["autostart_local_stream"] and _looks_like_local(self.cfg["camera_url"], self.cfg["stream_port"]):
            if not _is_stream_up(self.cfg["camera_url"]):
                LOG.info("Starting local stream...")
                _start_stream_background(port=self.cfg["stream_port"])
                _wait_until(lambda: _is_stream_up(self.cfg["camera_url"]), 8.0, 0.5)
        self.cap = cv2.VideoCapture(self.cfg["camera_url"])
        return self.cap.isOpened()

    def stop_all(self):
        try: self.car.set_velocity(0,0,0)
        except Exception: pass

    def run(self):
        LOG.info("🤖 Bruno Explorer started")
        if not self._open_camera(): LOG.warning("No camera available")
        self.running = True
        try:
            while self.running:
                self.frame_idx += 1; frame = None
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read(); frame = frame if ok else None

                d_cm = self.ultra.get_distance_cm()
                
                # ============ OBSTACLE AVOIDANCE & MOVEMENT ============
                if d_cm is not None and d_cm <= self.cfg["ultra_danger_cm"]:
                    # EMERGENCY STOP
                    self.ultra.set_rgb(255, 0, 0)  # red
                    self.stop_all()
                    current_action = f"EMERGENCY STOP ({d_cm:.1f}cm)"
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
                    current_action = f"ULTRA AVOID ({d_cm:.1f}cm) {'LEFT' if left else 'RIGHT'}"
                    LOG.info(current_action)
                    time.sleep(0.05)
                    continue
                else:
                    # SAFE TO MOVE FORWARD (or sensor reading failed)
                    self.ultra.set_rgb(0, 255, 0)  # green
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                    current_action = "FORWARD"
                    if d_cm is not None:
                        LOG.info(f"[ULTRA] {d_cm:.1f}cm - {current_action} (safe)")
                    else:
                        LOG.info(f"[ULTRA] No reading - {current_action} (assumed safe)")

                # ============ GPT VISION PHOTOS ============
                if frame is not None and self.vision_client.should_take_photo():
                    # Stop Bruno before taking picture
                    self.stop_all()
                    time.sleep(0.2)
                    LOG.info("📸 Taking GPT photo...")
                    
                    # Take photo and get GPT response
                    self.vision_client.capture_and_describe(frame, current_action)
                    
                    # Resume movement after GPT response
                    LOG.info("▶️ Resuming Bruno movement after photo...")
                    continue  # Skip to next iteration to resume normal movement
                
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        finally: self.shutdown()

    def shutdown(self):
        LOG.info("Shutting down..."); self.running = False; self.stop_all()
        if self.cap: self.cap.release()
        self.ultra.set_rgb(0,0,0)

# =========================
# Entrypoint
# =========================
RUNNER = None
def _sig_handler(signum, frame):
    global RUNNER
    if RUNNER: RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    RUNNER = BrunoIntelligentExplorer(CONFIG)
    RUNNER.run()
