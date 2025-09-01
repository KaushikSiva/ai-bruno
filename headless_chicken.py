#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Headless Obstacle Avoidance (Hiwonder RGB Ultrasonic + Optional Vision)
- Primary: Hiwonder ultrasonic distance -> immediate avoid/stop + RGB LED feedback
- Secondary: simple CV ROI check (optional; can be disabled)
- Headless safe (no GUI)
- Optional autostart of local MJPEG streamer (live_stream_test.py) if camera_url is localhost and down
"""

import os, sys, time, json, signal, logging, threading, urllib.request
from urllib.parse import urlparse
from typing import Optional, List, Dict

# --- Hiwonder SDK path ---
# Adjust if your MasterPi path differs
sys.path.append('/home/pi/MasterPi')

import cv2
import numpy as np

# Optional smoothing with pandas (fallback to mean if missing)
try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    pd = None
    PANDAS_OK = False

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

    # Debug
    "save_debug": False,
    "debug_interval": 15,                          # every N frames
}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("bruno.ultra")

# =========================
# Helpers: stream autostart
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
# Ultrasonic with LED
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
# Vision (optional): simple ROI + contour heuristic
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
# Main controller
# =========================
class BrunoAvoid:
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

    def stop_all(self):
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def run(self):
        LOG.info("Starting Bruno (ultrasonic primary, vision optional, headless)")
        if not self._open_camera():
            LOG.warning("Continuing without camera (ultrasonic-only mode).")
        self.running = True
        try:
            while self.running:
                self.frame_idx += 1
                frame = None
                if self.cap and self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if not ok: frame = None

                # --- Read ultrasonic and set LED ---
                d_cm = self.ultra.get_distance_cm()
                if d_cm is not None:
                    LOG.info(f"[ULTRA] {d_cm:.1f} cm")
                    if d_cm <= self.cfg["ultra_danger_cm"]:
                        self.ultra.set_rgb(255, 0, 0)            # red
                        self.stop_all()
                        msg = f"EMERGENCY STOP ({d_cm:.1f} cm)"
                        LOG.warning(msg)
                        self._save_debug(frame.copy() if frame is not None else np.zeros((240,320,3),np.uint8), msg, [])
                        time.sleep(0.05)
                        continue
                    elif d_cm <= self.cfg["ultra_caution_cm"]:
                        self.ultra.set_rgb(255, 180, 0)         # amber
                        # simple avoid: optional tiny backup then turn
                        if self.cfg["backup_time"] > 0:
                            self.car.set_velocity(self.cfg["turn_speed"], 90, 0)  # same style as sample; adjust if needed
                            time.sleep(self.cfg["backup_time"])
                            self.stop_all()
                        # alternate turn direction by frame blocks
                        left = ((self.frame_idx // 60) % 2 == 0)
                        self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                        time.sleep(self.cfg["turn_time"])
                        self.stop_all()
                        msg = f"ULTRA AVOID ({d_cm:.1f} cm) {'LEFT' if left else 'RIGHT'}"
                        self._save_debug(frame.copy() if frame is not None else np.zeros((240,320,3),np.uint8), msg, [])
                        continue
                    else:
                        self.ultra.set_rgb(0, 255, 0)            # green

                # --- Optional vision layer (secondary) ---
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
                            LOG.warning(msg)
                        elif px <= self.cfg["avoid_px"]:
                            left = closest["angle"] >= 0
                            self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                            time.sleep(self.cfg["turn_time"])
                            self.stop_all()
                            msg = f"VISION AVOID ({px:.1f}px) {'LEFT' if left else 'RIGHT'}"
                        else:
                            self.car.set_velocity(self.cfg["forward_speed"], 90, 0)
                            msg = "FORWARD (vision safe)"
                    else:
                        self.car.set_velocity(self.cfg["forward_speed"], 90, 0)  # same pattern as your example
                else:
                    # No frame or vision disabled: just go forward if ultrasonic says safe
                    self.car.set_velocity(self.cfg["forward_speed"], 90, 0)

                # Debug save (optional)
                if frame is not None:
                    self._save_debug(frame.copy(), msg, obs)

                time.sleep(0.03)  # loop rate limiter

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
    LOG.info("🤖 Bruno Headless (Hiwonder RGB Ultrasonic + Optional Vision)")
    LOG.info("Press Ctrl+C to stop")

    RUNNER = BrunoAvoid(CONFIG)
    RUNNER.run()
