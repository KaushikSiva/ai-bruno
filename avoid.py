#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Headless Avoid + Periodic GPT-Vision Bottle Check
- Ultrasonic (primary) + simple vision (optional) for roaming
- Every N seconds, pause -> capture -> ask GPT Vision if a bottle is visible
- If yes: turn toward it, approach to pickup distance, print READY FOR PICKUP, and exit
"""

import os, sys, time, json, signal, logging, threading, base64
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse
import urllib.request

# ---- Hiwonder SDK path (adjust if needed)
sys.path.append('/home/pi/MasterPi')

import cv2
import numpy as np

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

# OpenAI (GPT Vision)
try:
    from openai import OpenAI
    _openai_client = OpenAI()  # uses OPENAI_API_KEY env
    GPT_OK = True
except Exception as e:
    _openai_client = None
    GPT_OK = False

# =========================
# Config
# =========================
CONFIG = {
    # Camera
    "camera_url": os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream"),
    "autostart_local_stream": True,
    "stream_port": 8080,

    # Ultrasonic thresholds (cm)
    "ultra_caution_cm": float(os.environ.get("ULTRA_CAUTION_CM", "50")),
    "ultra_danger_cm":  float(os.environ.get("ULTRA_DANGER_CM",  "25")),

    # Drive params
    "forward_speed": int(os.environ.get("BRUNO_SPEED", "40")),
    "turn_speed":    int(os.environ.get("BRUNO_TURN_SPEED", "40")),
    "turn_time":     float(os.environ.get("BRUNO_TURN_TIME", "0.5")),
    "backup_time":   float(os.environ.get("BRUNO_BACKUP_TIME", "0.0")),

    # Vision (secondary)
    "use_vision": True,
    "roi": {"top": 0.5, "bottom": 0.95, "left": 0.15, "right": 0.85},
    "edge_min_area": 800,
    "danger_px": 70,
    "avoid_px": 110,

    # Debug
    "save_debug": False,
    "debug_interval": 15,

    # --- NEW: GPT Bottle Check ---
    "bottle_check_period_s": 30.0,             # how often to pause & check
    "gpt_model": os.environ.get("GPT_VISION_MODEL", "gpt-4o-mini"),
    "pickup_cm": float(os.environ.get("PICKUP_DISTANCE_CM", "15")),  # stop when <= this distance
    "approach_speed": int(os.environ.get("APPROACH_SPEED", "35")),
    "approach_timeout_s": float(os.environ.get("APPROACH_TIMEOUT_S", "30")),  # safety limit
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.ultra")

# =========================
# Stream helpers
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
# Ultrasonic + LED
# =========================
class UltrasonicRGB:
    def __init__(self, avg_samples: int = 5, sample_delay: float = 0.02):
        self.sonar = Sonar.Sonar()
        self.avg_samples = avg_samples
        self.sample_delay = sample_delay
        self.set_rgb(0, 0, 255)  # heartbeat blue

    def get_distance_cm(self) -> Optional[float]:
        vals: List[float] = []
        for _ in range(self.avg_samples):
            try:
                raw = self.sonar.getDistance()
                d_cm = (raw / 10.0) if raw > 100 else float(raw)  # mm->cm autodetect
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
        for idx in (0, 1):
            try:
                self.sonar.setPixelColor(idx, (r, g, b))
            except TypeError:
                try:
                    self.sonar.setPixelColor(idx, r, g, b)
                except Exception:
                    pass
            except Exception:
                pass

# =========================
# Optional simple vision
# =========================
def _estimate_px_distance(w: int, h: int, y: int, H: int) -> float:
    size = (w * h) / (100 * 100)
    pos  = (H - y) / max(H, 1)
    prox = size * 0.6 + pos * 0.4
    return max(20, 200 - prox * 100)

def _vision_obstacles(frame: np.ndarray, cfg: Dict) -> List[Dict]:
    try:
        H, W = frame.shape[:2]
        r = cfg["roi"]; y1, y2 = int(H*r["top"]), int(H*r["bottom"]); x1, x2 = int(W*r["left"]), int(W*r["right"])
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
# GPT Vision bottle check
# =========================
def _encode_frame_to_data_url(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _extract_json(s: str) -> Dict:
    # Robustly extract the first {...} JSON object
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(s[i:j+1])
    # Fallback if already raw JSON
    return json.loads(s)

def gpt_bottle_check(frame: np.ndarray, model: str) -> Tuple[bool, str]:
    """
    Returns (is_bottle, position) where position in {'left','center','right','unknown'}
    """
    if not GPT_OK:
        LOG.warning("OpenAI not available; skipping GPT bottle check.")
        return (False, "unknown")

    data_url = _encode_frame_to_data_url(frame)
    prompt = (
        "You are a vision classifier. Determine if a standard plastic WATER BOTTLE is visible. "
        "Return STRICT one-line JSON only: "
        '{"is_bottle": true|false, "position": "left|center|right|unknown"} '
        "Position is where the bottle mainly appears relative to the image center. "
        "If unsure, return false."
    )

    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ],
            }],
        )
        out = resp.choices[0].message.content.strip()
        obj = _extract_json(out)
        is_bottle = bool(obj.get("is_bottle", False))
        pos = str(obj.get("position", "unknown")).lower()
        if pos not in ("left", "center", "right", "unknown"):
            pos = "unknown"
        return (is_bottle, pos)
    except Exception as e:
        LOG.warning(f"GPT bottle check failed: {e}")
        return (False, "unknown")

# =========================
# Main controller
# =========================
class BrunoAvoid:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.car = mecanum.MecanumChassis()
        self.board = Board()
        self.AK = ArmIK(); self.AK.board = self.board

        # Arm/gripper like sample
        try:
            servo1 = 1500
            self.board.pwm_servo_set_position(0.3, [[1, servo1]])
            self.AK.setPitchRangeMoving((0, 6, 18), 0, -90, 90, 1500)
        except Exception as e:
            LOG.warning(f"Arm init skipped: {e}")

        self.ultra = UltrasonicRGB()
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_idx = 0
        self._last_check = time.monotonic()  # for 30s cadence

    # ---- motion wrappers for API variants
    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            try:
                self.car.set_velocity(0, 90, 0)
            except Exception:
                pass

    def drive_forward(self, speed: int):
        try:
            self.car.set_velocity(speed, 0, 0)         # vx,vy,vw
        except Exception:
            self.car.set_velocity(speed, 90, 0)        # speed,angle,omega

    def drive_turn(self, left: bool, omega=0.5, dur=0.5):
        try:
            self.car.set_velocity(0, 0, (-omega if left else omega))
        except Exception:
            self.car.set_velocity(0, 90, (-omega if left else omega))
        time.sleep(dur)
        self.stop_all()

    # ---- camera handling
    def _autostart_stream_if_needed(self):
        cam_url = self.cfg["camera_url"]
        if self.cfg["autostart_local_stream"] and _looks_like_local(cam_url, self.cfg["stream_port"]):
            if not _is_stream_up(cam_url):
                LOG.info(f"Local stream {cam_url} not running. Launching live_stream_test...")
                _start_stream_background(host="0.0.0.0", port=self.cfg["stream_port"])
                _wait_until(lambda: _is_stream_up(cam_url), timeout_s=8.0, interval_s=0.5)

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
            try:
                # reduce latency for USB cams
                if isinstance(src, int) or str(src).isdigit():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            return True
        LOG.error(f"Failed to open camera source: {src}")
        return False

    def _grab_frame(self) -> Optional[np.ndarray]:
        if not (self.cap and self.cap.isOpened()):
            if not self._open_camera():
                return None
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    # ---- NEW: periodic bottle check + approach
    def _maybe_do_bottle_check(self) -> bool:
        """Returns True if we reached pickup and should exit."""
        period = self.cfg["bottle_check_period_s"]
        if time.monotonic() - self._last_check < period:
            return False

        self._last_check = time.monotonic()
        LOG.info("⏸️  Periodic check: pausing to take a photo for GPT…")
        self.stop_all()
        time.sleep(0.2)

        frame = self._grab_frame()
        if frame is None:
            LOG.warning("No frame for GPT check; skipping this cycle.")
            return False

        is_bottle, pos = gpt_bottle_check(frame, self.cfg["gpt_model"])
        LOG.info(f"GPT says bottle={is_bottle}, position={pos}")

        if not is_bottle:
            return False

        # Turn toward position once, then approach using ultrasonic
        if pos == "left":
            self.drive_turn(left=True, omega=0.6, dur=self.cfg["turn_time"])
        elif pos == "right":
            self.drive_turn(left=False, omega=0.6, dur=self.cfg["turn_time"])

        return self._approach_to_pickup()

    def _approach_to_pickup(self) -> bool:
        """Drive forward until ultrasonic <= pickup_cm, then print and stop. Returns True if succeeded."""
        target = float(self.cfg["pickup_cm"])
        speed  = int(self.cfg["approach_speed"])
        t0     = time.monotonic()
        LOG.info(f"🚶 Approaching bottle until distance <= {target:.1f} cm …")

        while time.monotonic() - t0 < self.cfg["approach_timeout_s"]:
            d = self.ultra.get_distance_cm()
            if d is not None:
                LOG.info(f"[APPROACH] {d:.1f} cm")
                if d <= target:
                    self.stop_all()
                    print("READY FOR PICKUP")  # <-- required message
                    return True
                # gentle crawl when close
                if d < max(target * 2.0, 30):
                    self.drive_forward(max(20, speed - 10))
                else:
                    self.drive_forward(speed)
            else:
                # if no reading, creep slowly but safely
                self.drive_forward(max(15, speed - 15))
            time.sleep(0.05)

        LOG.warning("Approach timeout; did not reach pickup distance.")
        self.stop_all()
        return False

    # ---- debug save (unchanged)
    def _save_debug(self, frame: np.ndarray, msg: str, obs: List[Dict]):
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
            for o in obs:
                x,y,w,h = o["bbox"]; dpx = o["px"]
                color = (0,0,255) if dpx <= self.cfg["danger_px"] else ((0,255,255) if dpx <= self.cfg["avoid_px"] else (0,255,0))
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{dpx:.1f}px", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, msg, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"debug_images/debug_{ts}_{self.frame_idx:06d}.jpg"
            cv2.imwrite(fn, frame)
            LOG.info(f"Saved {fn}")
        except Exception as e:
            LOG.warning(f"Save debug failed: {e}")

    # ---- main loop
    def run(self):
        LOG.info("Starting Bruno (ultrasonic primary, vision optional, headless)")
        if not self._open_camera():
            LOG.warning("Continuing without camera (ultrasonic-only mode).")
        self.running = True
        try:
            while self.running:
                self.frame_idx += 1

                # Periodic GPT bottle check
                if self._maybe_do_bottle_check():
                    # reached pickup -> graceful stop
                    break

                # Normal roam: read ultrasonic, simple avoid, optional vision
                frame = None
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok: frame = None

                d_cm = self.ultra.get_distance_cm()
                if d_cm is not None:
                    LOG.info(f"[ULTRA] {d_cm:.1f} cm")
                    if d_cm <= self.cfg["ultra_danger_cm"]:
                        self.ultra.set_rgb(255, 0, 0)
                        self.stop_all()
                        time.sleep(0.05)
                        continue
                    elif d_cm <= self.cfg["ultra_caution_cm"]:
                        self.ultra.set_rgb(255, 180, 0)
                        if self.cfg["backup_time"] > 0:
                            self.drive_forward(-self.cfg["turn_speed"])  # tiny nudge back if your API supports negative vx
                            time.sleep(self.cfg["backup_time"])
                            self.stop_all()
                        left = ((self.frame_idx // 60) % 2 == 0)
                        self.drive_turn(left=left, omega=0.5, dur=self.cfg["turn_time"])
                        continue
                    else:
                        self.ultra.set_rgb(0, 255, 0)

                msg = "FORWARD (ultra safe)"
                obs = []
                if frame is not None and self.cfg["use_vision"]:
                    obs = _vision_obstacles(frame, self.cfg)
                    if obs:
                        closest = min(obs, key=lambda o: o["px"])
                        px = closest["px"]
                        if px <= self.cfg["danger_px"]:
                            self.stop_all()
                            msg = f"VISION STOP ({px:.1f}px)"
                        elif px <= self.cfg["avoid_px"]:
                            left = closest["angle"] >= 0
                            self.drive_turn(left=left, omega=0.5, dur=self.cfg["turn_time"])
                            msg = f"VISION AVOID ({px:.1f}px) {'LEFT' if left else 'RIGHT'}"
                        else:
                            self.drive_forward(self.cfg["forward_speed"])
                            msg = "FORWARD (vision safe)"
                    else:
                        self.drive_forward(self.cfg["forward_speed"])
                else:
                    self.drive_forward(self.cfg["forward_speed"])

                if frame is not None:
                    self._save_debug(frame.copy(), msg, obs)

                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        LOG.info("Shutting down…")
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
RUNNER: Optional[BrunoAvoid] = None

def _sig_handler(signum, frame):
    global RUNNER
    print("\n🛑 Ctrl-C received; stopping Bruno...")
    if RUNNER:
        RUNNER.shutdown()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig_handler)
    LOG.info("🤖 Bruno Headless (with periodic GPT bottle check)")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoAvoid(CONFIG)
    RUNNER.run()
