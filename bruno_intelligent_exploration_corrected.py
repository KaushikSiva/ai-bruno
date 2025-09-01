#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Intelligent Exploration (Based on headless_chicken.py)
- Primary: Hiwonder ultrasonic distance -> immediate avoid/stop + RGB LED feedback
- Secondary: Optional vision detection
- GPT Vision: Takes photo every 20 seconds and gets AI descriptions
- Headless safe (no GUI)
"""

import os, sys, time, json, signal, logging, threading, urllib.request
import base64, io
from urllib.parse import urlparse
from typing import Optional, List, Dict

# --- Hiwonder SDK path ---
# Adjust if your MasterPi path differs
sys.path.append('/home/pi/MasterPi')

import cv2
import numpy as np
from PIL import Image

# Optional smoothing with pandas (fallback to mean if missing)
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
except ImportError:
    print("Warning: OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Hiwonder modules
import common.sonar as Sonar               # RGB ultrasonic
import common.mecanum as mecanum           # mecanum chassis
from common.ros_robot_controller_sdk import Board
from kinematics.transform import *         # used by ArmIK stack
from kinematics.arm_move_ik import ArmIK   # arm init like the working sample

# =========================
# Config
# =========================
CONFIG = {
    # Camera stream
    "camera_url": os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream"),
    "autostart_local_stream": True,
    "stream_port": 8080,

    # Ultrasonic thresholds (cm)
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),  # start avoiding
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM", "25")),   # emergency stop

    # Drive parameters
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),  # same baseline as your sample
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),   # set >0 if you want a small backup before turning

    # Vision (optional)
    "use_vision": True,                            # set False to rely on ultrasonic only
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,                          # contour min area
    "danger_px": 70,                               # smaller = closer (pixel heuristic)
    "avoid_px": 110,

    # GPT Vision settings
    "gpt_photo_interval": 20,                      # Take photo every 20 seconds
    "gpt_vision_prompt": "Describe what you see in this image. Focus on: objects, furniture, rooms, people, pets, bottles, bins/containers, obstacles, walls, and the general environment. Be concise but detailed.",
    "save_gpt_images": True,                       # Save images sent to GPT
    
    # Debug
    "save_debug": True,
    "debug_interval": 30,                          # every N frames
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.intelligent")

# =========================
# Helpers: stream autostart (from headless_chicken.py)
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
# Ultrasonic with LED (from headless_chicken.py)
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
                # Hiwonder returns mm or cm depending on build; sample code divides by 10 to get cm.
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
        # Some builds accept led index 0 for both; also set 1 and 2 explicitly
        try:
            self.sonar.setPixelColor(0, (r, g, b))
        except Exception:
            pass
        try:
            self.sonar.setPixelColor(1, (r, g, b))
        except Exception:
            pass

# =========================
# Vision (optional): simple ROI + contour heuristic (from headless_chicken.py)
# =========================
def _estimate_px_distance(w: int, h: int, y: int, H: int) -> float:
    # larger + lower in frame => closer (smaller returned distance)
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
# GPT Vision Integration
# =========================
class GPTVision:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.enabled = False
        self.client = None
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
            self.enabled = True
            LOG.info("✓ GPT Vision enabled")
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                LOG.warning("⚠️  OPENAI_API_KEY not set - GPT Vision disabled")
            else:
                LOG.warning("⚠️  OpenAI library not available - GPT Vision disabled")
        
        # GPT tracking
        self.last_photo_time = 0
        self.photo_count = 0
        self.descriptions = []
    
    def should_take_photo(self) -> bool:
        """Check if it's time for a GPT photo"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        return (current_time - self.last_photo_time) >= self.cfg["gpt_photo_interval"]
    
    def capture_and_describe(self, frame: np.ndarray, current_action: str = "UNKNOWN") -> Optional[str]:
        """Capture frame, send to GPT Vision, return description"""
        if not self.enabled:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Save image if configured
            if self.cfg.get("save_gpt_images", True):
                os.makedirs("gpt_images", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gpt_images/bruno_explore_{timestamp}_{self.photo_count:04d}.jpg"
                image.save(filename)
                LOG.info(f"📸 GPT image saved: {filename}")
            
            # Encode for GPT
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            # Ask GPT Vision
            LOG.info("🧠 Asking GPT Vision for description...")
            
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
            
            # Store description with metadata
            self.photo_count += 1
            self.last_photo_time = time.time()
            
            description_entry = {
                'timestamp': time.strftime("%H:%M:%S"),
                'photo_count': self.photo_count,
                'description': description,
                'current_action': current_action
            }
            self.descriptions.append(description_entry)
            
            # Log the description
            LOG.info("\n" + "=" * 60)
            LOG.info("🔍 GPT VISION DESCRIPTION")
            LOG.info("=" * 60)
            LOG.info(f"Time: {description_entry['timestamp']} | Photo: #{self.photo_count} | Action: {current_action}")
            LOG.info(description)
            LOG.info("=" * 60)
            
            return description
            
        except APIError as e:
            LOG.error(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            LOG.error(f"GPT Vision error: {e}")
            return None
    
    def get_next_photo_countdown(self) -> float:
        """Get seconds until next photo"""
        if not self.enabled:
            return float('inf')
        
        elapsed = time.time() - self.last_photo_time
        return max(0, self.cfg["gpt_photo_interval"] - elapsed)

# =========================
# Main controller (enhanced from headless_chicken.py)
# =========================
class BrunoIntelligentExplorer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        # Arm/gripper like the working sample
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
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'ultrasonic_readings': 0,
            'emergency_stops': 0,
            'ultrasonic_avoids': 0,
            'vision_avoids': 0,
            'gpt_photos_taken': 0,
            'gpt_descriptions_received': 0,
        }
        
        self.start_time = time.time()

    def _autostart_stream_if_needed(self):
        cam_url = self.cfg["camera_url"]
        if self.cfg["autostart_local_stream"] and _looks_like_local(cam_url, self.cfg["stream_port"]):
            if not _is_stream_up(cam_url):
                LOG.info(f"Local stream {cam_url} not running. Launching live_stream_test...")
                _start_stream_background(host="0.0.0.0", port=self.cfg["stream_port"])
                ok = _wait_until(lambda: _is_stream_up(cam_url), timeout_s=8.0, interval_s=0.5)
                if not ok:
                    LOG.warning("Local stream did not come up in time (continuing; camera may still open later).")

    def _open_camera(self) -> bool:
        self._autostart_stream_if_needed()
        src = self.cfg["camera_url"]
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            # try ffmpeg backend if available
            try:
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            except Exception:
                pass
        if self.cap and self.cap.isOpened():
            return True
        LOG.error(f"Failed to open camera source: {src}")
        return False

    def _save_debug(self, frame: np.ndarray, msg: str, obs: List[Dict], ultrasonic_cm: Optional[float] = None):
        if not self.cfg["save_debug"]:
            return
        if self.frame_idx % self.cfg["debug_interval"] != 0:
            return
        try:
            os.makedirs("debug_images", exist_ok=True)
            H, W = frame.shape[:2]
            r = self.cfg["roi"]
            y1, y2 = int(H*r["top"]), int(H*r["bottom"])
            x1, x2 = int(W*r["left"]), int(W*r["right"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
            
            # Draw vision obstacles
            for o in obs:
                x,y,w,h = o["bbox"]; dpx = o["px"]
                color = (0,0,255) if dpx <= self.cfg["danger_px"] else ((0,255,255) if dpx <= self.cfg["avoid_px"] else (0,255,0))
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{dpx:.1f}px", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add action and ultrasonic info
            cv2.putText(frame, msg, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
            if ultrasonic_cm is not None:
                ultra_text = f"Ultrasonic: {ultrasonic_cm:.1f}cm"
                cv2.putText(frame, ultra_text, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            
            # Add GPT Vision status
            if self.gpt_vision.enabled:
                next_photo = self.gpt_vision.get_next_photo_countdown()
                gpt_text = f"Next GPT photo: {next_photo:.1f}s | Photos: {self.gpt_vision.photo_count}"
                cv2.putText(frame, gpt_text, (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"debug_images/intelligent_debug_{ts}_{self.frame_idx:06d}.jpg"
            cv2.imwrite(fn, frame)
        except Exception as e:
            LOG.warning(f"Save debug failed: {e}")

    def _print_status_report(self):
        """Print periodic status report"""
        if self.frame_idx % 200 != 0:  # Every 200 frames
            return
        
        elapsed_time = time.time() - self.start_time
        fps = self.frame_idx / elapsed_time if elapsed_time > 0 else 0
        
        LOG.info("=" * 70)
        LOG.info("📊 BRUNO INTELLIGENT EXPLORATION STATUS")
        LOG.info("=" * 70)
        LOG.info(f"Runtime: {elapsed_time:.1f}s | FPS: {fps:.1f} | Frames: {self.frame_idx}")
        LOG.info(f"Ultrasonic readings: {self.stats['ultrasonic_readings']}")
        LOG.info(f"Emergency stops: {self.stats['emergency_stops']} | Ultrasonic avoids: {self.stats['ultrasonic_avoids']}")
        LOG.info(f"Vision avoids: {self.stats['vision_avoids']}")
        if self.gpt_vision.enabled:
            LOG.info(f"GPT Photos: {self.gpt_vision.photo_count} | Descriptions received: {len(self.gpt_vision.descriptions)}")
            next_photo = self.gpt_vision.get_next_photo_countdown()
            LOG.info(f"Next GPT photo in: {next_photo:.1f}s")
        else:
            LOG.info("GPT Vision: Disabled")
        LOG.info("=" * 70)

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def save_exploration_log(self):
        """Save complete exploration session log"""
        try:
            os.makedirs("exploration_logs", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"exploration_logs/intelligent_exploration_{timestamp}.json"
            
            session_data = {
                'session_start': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                'session_end': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_runtime': time.time() - self.start_time,
                'statistics': self.stats,
                'config': self.cfg,
                'gpt_descriptions': self.gpt_vision.descriptions if self.gpt_vision.enabled else []
            }
            
            with open(log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            LOG.info(f"📝 Exploration log saved: {log_file}")
            
        except Exception as e:
            LOG.error(f"Error saving exploration log: {e}")

    def run(self):
        LOG.info("🤖 Starting Bruno Intelligent Explorer (ultrasonic primary, vision optional, GPT descriptions)")
        if self.gpt_vision.enabled:
            LOG.info(f"📸 GPT Vision enabled - photos every {self.cfg['gpt_photo_interval']}s")
        else:
            LOG.info("📸 GPT Vision disabled")
        
        if not self._open_camera():
            LOG.warning("Continuing without camera (ultrasonic-only mode).")
        
        self.running = True
        current_action = "STARTING"
        
        try:
            while self.running:
                self.frame_idx += 1
                self.stats['frames_processed'] = self.frame_idx
                frame = None
                
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok: 
                        frame = None

                # --- Primary: Read ultrasonic and set LED (from headless_chicken.py) ---
                d_cm = self.ultra.get_distance_cm()
                if d_cm is not None:
                    self.stats['ultrasonic_readings'] += 1
                    LOG.info(f"[ULTRA] {d_cm:.1f} cm")
                    
                    if d_cm <= self.cfg["ultra_danger_cm"]:
                        self.ultra.set_rgb(255, 0, 0)            # red
                        self.stop_all()
                        current_action = f"EMERGENCY STOP ({d_cm:.1f} cm)"
                        LOG.warning(current_action)
                        self.stats['emergency_stops'] += 1
                        self._save_debug(frame.copy() if frame is not None else np.zeros((240,320,3),np.uint8), current_action, [], d_cm)
                        time.sleep(0.05)
                        continue
                        
                    elif d_cm <= self.cfg["ultra_caution_cm"]:
                        self.ultra.set_rgb(255, 180, 0)         # amber
                        # simple avoid: optional tiny backup then turn
                        if self.cfg["backup_time"] > 0:
                            self.car.set_velocity(self.cfg["turn_speed"], 90, 0)  # backup
                            time.sleep(self.cfg["backup_time"])
                            self.stop_all()
                        # alternate turn direction by frame blocks
                        left = ((self.frame_idx // 60) % 2 == 0)
                        self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                        time.sleep(self.cfg["turn_time"])
                        self.stop_all()
                        current_action = f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}"
                        LOG.info(current_action)
                        self.stats['ultrasonic_avoids'] += 1
                        self._save_debug(frame.copy() if frame is not None else np.zeros((240,320,3),np.uint8), current_action, [], d_cm)
                        continue
                    else:
                        self.ultra.set_rgb(0, 255, 0)            # green

                # --- Secondary: Optional vision layer ---
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
                            self.stats['vision_avoids'] += 1
                        else:
                            self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                            current_action = "FORWARD (vision safe)"
                    else:
                        self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                else:
                    # No frame or vision disabled: just go forward if ultrasonic says safe
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # --- GPT Vision: Take photo and describe every 20 seconds ---
                if frame is not None and self.gpt_vision.should_take_photo():
                    LOG.info(f"📸 Time for GPT photo! (interval: {self.cfg['gpt_photo_interval']}s)")
                    description = self.gpt_vision.capture_and_describe(frame, current_action)
                    if description:
                        self.stats['gpt_descriptions_received'] += 1

                # Debug save (optional)
                if frame is not None:
                    self._save_debug(frame.copy(), current_action, obs, d_cm)

                # Status report
                self._print_status_report()

                time.sleep(0.03)  # loop rate limiter (same as headless_chicken.py)

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
        
        # turn LEDs off
        try:
            self.ultra.set_rgb(0, 0, 0)
        except Exception:
            pass
        
        # Save exploration log
        self.save_exploration_log()

# =========================
# Entrypoint
# =========================
RUNNER: Optional[BrunoIntelligentExplorer] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping Bruno Intelligent Explorer...")
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno Intelligent Explorer (Hiwonder RGB Ultrasonic + Vision + GPT)")
    LOG.info("📸 Takes GPT Vision photos every 20 seconds while navigating safely")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoIntelligentExplorer(CONFIG)
    RUNNER.run()