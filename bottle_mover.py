#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Bottle Picker (Upright + Visual Servoing PD)
- Patrols: roam, snapshot, caption via captioner.py
- On bottle caption: APPROACH mode with visual servoing to center the bottle
  * Tracker produces horizontal offset in [-1,+1]
  * PD yaw controller keeps bottle centered while advancing
  * Ultrasonic enforces safety and final stop distance
- When within APPROACH_STOP_CM: side-grip pick (ArmIK + servo), then exit

Auto-creates ./gpt_images, ./logs, ./debug
Logs to STDOUT and ./logs/bruno.log
"""

import os, sys, time, signal, logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from logging.handlers import RotatingFileHandler

# =========================
# Logging & folders
# =========================
def setup_logging_to_stdout(name: str = "bruno.bottle") -> logging.Logger:
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

def ensure_dirs() -> Dict[str, Path]:
    base = Path.cwd()
    paths = {
        "base": base,
        "images": base / "gpt_images",
        "logs": base / "logs",
        "debug": base / "debug",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

LOG = setup_logging_to_stdout("bruno.bottle")
paths = ensure_dirs()

def save_image_path(prefix: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    us = int((time.time() * 1_000_000) % 1_000_000)
    return paths["images"] / f"{prefix}_{ts}_{us:06d}.jpg"

# =========================
# Env knobs (tunable)
# =========================
PHOTO_INTERVAL        = int(os.environ.get("PHOTO_INTERVAL_SEC", "8"))

# Ultrasonic thresholds (cm)
ULTRA_CAUTION_CM      = float(os.environ.get("ULTRA_CAUTION_CM", "50"))
ULTRA_DANGER_CM       = float(os.environ.get("ULTRA_DANGER_CM", "25"))
APPROACH_STOP_CM      = float(os.environ.get("APPROACH_STOP_CM", "22"))
APPROACH_MAX_TIME_S   = float(os.environ.get("APPROACH_MAX_TIME_S", "60"))

# Drive/turn
FORWARD_SPEED         = int(os.environ.get("BRUNO_SPEED", "40"))
TURN_TIME             = float(os.environ.get("BRUNO_TURN_TIME", "0.5"))

# Visual servoing gains
VS_KP                 = float(os.environ.get("VS_KP", "0.9"))   # proportional yaw gain
VS_KD                 = float(os.environ.get("VS_KD", "0.15"))  # derivative yaw gain
VS_YAW_LIMIT          = float(os.environ.get("VS_YAW_LIMIT", "0.9"))  # clamp for stability

# Caption detection
BOTTLE_KEYWORDS = [s.strip().lower() for s in os.environ.get(
    "BOTTLE_KEYWORDS",
    "water bottle,plastic bottle,bottle,bottle of water,drinking bottle"
).split(",")]

# Arm/gripper
GRIPPER_SERVO_ID      = int(os.environ.get("GRIPPER_SERVO_ID", "1"))
GRIP_OPEN             = int(os.environ.get("GRIP_OPEN", "1600"))
GRIP_CLOSE            = int(os.environ.get("GRIP_CLOSE", "1100"))
GRIP_TIME             = float(os.environ.get("GRIP_TIME", "0.35"))  # s

# Bottle geometry / grasp height
BOTTLE_HEIGHT_CM      = float(os.environ.get("BOTTLE_HEIGHT_CM", "20"))
GRIP_AT_RATIO         = float(os.environ.get("GRIP_AT_RATIO", "0.45"))
GRIP_Z_COMPUTED       = max(6.0, min(BOTTLE_HEIGHT_CM * GRIP_AT_RATIO, 14.0))

# IK poses (cm) & pitch (deg)
PREGRASP_X            = float(os.environ.get("PREGRASP_X", "12"))
PREGRASP_Y            = float(os.environ.get("PREGRASP_Y", "0"))
PREGRASP_Z            = float(os.environ.get("PREGRASP_Z", "18"))
GRASP_Z               = float(os.environ.get("GRASP_Z", str(GRIP_Z_COMPUTED)))
LIFT_Z                = float(os.environ.get("LIFT_Z", "22"))
IK_ALPHA              = float(os.environ.get("IK_ALPHA", "-8"))
IK_A1                 = float(os.environ.get("IK_A1", "-90"))
IK_A2                 = float(os.environ.get("IK_A2", "90"))
IK_TIME_FAST          = int(os.environ.get("IK_TIME_FAST", "600"))
IK_TIME_SLOW          = int(os.environ.get("IK_TIME_SLOW", "1200"))

# =========================
# SDK & deps
# =========================
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

import captioner  # your local HF pipeline captioner

# Hiwonder modules
import common.sonar as Sonar
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board
from kinematics.arm_move_ik import ArmIK

# =========================
# Utilities
# =========================
def looks_like_bottle(text: str) -> bool:
    c = (text or "").lower()
    return any(k in c for k in BOTTLE_KEYWORDS)

def sidecar_txt(p: Path) -> Path:
    return p.with_suffix(".txt")

# =========================
# Ultrasonic helper
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
            s = pd.Series(vals); m, std = float(s.mean()), float(s.std())
            if std > 0:
                s = s[np.abs(s - m) <= std]
            if len(s) > 0:
                return float(s.mean())
        return float(np.mean(vals))

    def set_rgb(self, r: int, g: int, b: int):
        for i in (0,1):
            try: self.sonar.setPixelColor(i, (r,g,b))
            except Exception: pass

# =========================
# Camera helpers
# =========================
def list_cams() -> List[Dict]:
    cams=[]
    for i in range(8):
        dev=f"/dev/video{i}"
        if os.path.exists(dev):
            cams.append({"path":dev,"index":i,"accessible":os.access(dev, os.R_OK)})
    return cams

def open_cam(device_info: Dict) -> Optional[cv2.VideoCapture]:
    dp = device_info["path"]; idx = device_info["index"]
    for name, fn in [
        ("V4L2 idx", lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ("V4L2 path", lambda: cv2.VideoCapture(dp, cv2.CAP_V4L2)),
        ("Default idx", lambda: cv2.VideoCapture(idx)),
        ("Default path", lambda: cv2.VideoCapture(dp)),
    ]:
        cap = fn()
        if cap and cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                LOG.info(f"✅ Camera {dp} via {name}")
                return cap
            cap.release()
    return None

def find_cam() -> Optional[cv2.VideoCapture]:
    cams = list_cams()
    for c in cams:
        if not c["accessible"]: continue
        cap = open_cam(c)
        if cap: return cap
    LOG.error("❌ No /dev/video* camera available")
    return None

# =========================
# Bottle Tracker (Hiwonder hook + fallback)
# =========================
class BottleTracker:
    """
    Produces a normalized horizontal offset in [-1,+1]:
      -1 = far left, +1 = far right, 0 = centered
    First tries to use a Hiwonder target tracker module if available.
    Otherwise uses a simple contour-based fallback in a bottom ROI.
    """
    def __init__(self, img_w: int = 1280, img_h: int = 720):
        self.img_w = img_w
        self.img_h = img_h
        self.prev_err = 0.0

        # Attempt to import a Hiwonder tracker (you can adapt this to your actual module)
        self.use_hw_tracker = False
        try:
            # Example: from docs target tracking — replace with your module/class if present
            # from target_tracking import TargetTracker
            # self.hw_tracker = TargetTracker()
            # self.use_hw_tracker = True
            pass
        except Exception:
            self.use_hw_tracker = False

        # Fallback ROI (bottom center band)
        self.roi_top = int(self.img_h * 0.50)
        self.roi_bottom = int(self.img_h * 0.95)
        self.roi_left = int(self.img_w * 0.15)
        self.roi_right = int(self.img_w * 0.85)

    def _fallback_offset(self, frame: np.ndarray) -> Optional[float]:
        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        edges = cv2.Canny(g, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0.0
        H, W = roi.shape[:2]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w*h
            if area < 800:  # ignore tiny
                continue
            aspect = h / max(w,1)
            # bottle-like: tall-ish rectangle
            if aspect < 1.4:
                continue
            # favor larger + more central detections
            cx = x + w/2
            center_weight = 1.0 - abs(cx - W/2) / (W/2)
            score = area * (0.5 + 0.5*center_weight)
            if score > best_score:
                best_score = score
                best = (cx, w, h)

        if best is None:
            return None
        cx, _, _ = best
        # map ROI center offset to [-1,1]
        err_pixels = (cx - W/2)
        norm = float(err_pixels) / (W/2)
        return float(np.clip(norm, -1.0, 1.0))

    def offset(self, frame: np.ndarray) -> Optional[float]:
        # If you have a real Hiwonder tracker, call it here to get an offset:
        if self.use_hw_tracker:
            try:
                # norm = self.hw_tracker.horizontal_offset(frame)  # expected [-1,1]
                # return float(np.clip(norm, -1.0, 1.0))
                pass
            except Exception:
                pass
        # Fallback
        return self._fallback_offset(frame)

# =========================
# Main class
# =========================
class BrunoBottlePickerVS:
    def __init__(self):
        self.car   = mecanum.MecanumChassis()
        self.board = Board()
        self.AK    = ArmIK(); self.AK.board = self.board
        self.ultra = UltrasonicRGB()
        self.cap: Optional[cv2.VideoCapture] = None

        self.running = False
        self.frame_idx = 0

        self.snapshot_last_t = 0.0
        self.bottle_spotted = False
        self.approach_started_at: Optional[float] = None

        self.tracker = BottleTracker()

        # PD state
        self.err_prev = 0.0
        self.t_prev = time.time()

    # --- Gripper helpers ---
    def gripper_open(self):
        self.board.pwm_servo_set_position(0.3, [[GRIPPER_SERVO_ID, GRIP_OPEN]])
        time.sleep(GRIP_TIME)

    def gripper_close(self):
        self.board.pwm_servo_set_position(0.3, [[GRIPPER_SERVO_ID, GRIP_CLOSE]])
        time.sleep(GRIP_TIME)

    # --- IK wrapper ---
    def ik_move(self, xyz, ms=800):
        return self.AK.setPitchRangeMoving(xyz, IK_ALPHA, IK_A1, IK_A2, int(ms))

    # --- Patrol snapshot/caption ---
    def maybe_snapshot_and_caption(self, frame) -> Optional[str]:
        now = time.time()
        if self.snapshot_last_t == 0 or (now - self.snapshot_last_t) >= PHOTO_INTERVAL:
            self.snapshot_last_t = now
            self.car.set_velocity(0,0,0); time.sleep(0.10)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            img_path = save_image_path("bottle_patrol")
            image.save(str(img_path))
            LOG.info(f"💾 Snapshot saved: {img_path}")
            caption = captioner.get_caption(str(img_path))
            LOG.info(f"📝 Caption: {caption}")
            try:
                sidecar_txt(img_path).write_text(caption+"\n", encoding="utf-8")
            except Exception:
                pass
            return caption
        return None

    # --- Visual servoing approach ---
    def approach_step(self, frame: np.ndarray):
        if self.approach_started_at is None:
            self.approach_started_at = time.time()
            self.err_prev = 0.0
            self.t_prev = time.time()
            LOG.info("🚶 APPROACH (VS) engaged…")

        # Timeout guard
        if (time.time() - self.approach_started_at) > APPROACH_MAX_TIME_S:
            LOG.warning("⏳ Approach timeout — stopping.")
            self.shutdown(); sys.exit(0)

        d = self.ultra.get_distance_cm()

        # Safety: emergency range
        if d is not None and d <= ULTRA_DANGER_CM:
            self.ultra.set_rgb(255,0,0)
            self.car.set_velocity(0,0,0); time.sleep(0.05)
            # brief reverse + slight left to unstick
            self.car.set_velocity(-FORWARD_SPEED//2, 90, 0); time.sleep(0.35)
            self.car.set_velocity(0, 90, -0.7); time.sleep(0.30)
            self.car.set_velocity(0,0,0); time.sleep(0.05)
            return

        # Close enough → pick
        if d is not None and d <= APPROACH_STOP_CM:
            self.ultra.set_rgb(0,0,255)
            self.car.set_velocity(0,0,0)
            LOG.info(f"✅ Reached bottle (~{d:.1f} cm). Starting PICK sequence…")
            self.pick_sequence_upright()
            LOG.info("🛑 Pick complete. Exiting.")
            self.shutdown(); sys.exit(0)

        # Compute offset and PD yaw
        err = self.tracker.offset(frame)
        if err is None:
            # No visual cue; gentle scan so we keep searching
            phase = (time.time() % 1.2) / 1.2
            yaw = -0.4 if phase < 0.5 else 0.4
            self.ultra.set_rgb(0, 200, 255)
            self.car.set_velocity(FORWARD_SPEED//2, 90, yaw)
            return

        # PD
        t_now = time.time()
        dt = max(1e-3, t_now - self.t_prev)
        derr = (err - self.err_prev) / dt
        yaw = VS_KP * err + VS_KD * derr
        yaw = float(np.clip(yaw, -VS_YAW_LIMIT, VS_YAW_LIMIT))

        self.err_prev = err
        self.t_prev = t_now

        # Advance with PD yaw
        self.ultra.set_rgb(0,255,0)
        self.car.set_velocity(FORWARD_SPEED//2, 90, yaw)

    # --- Upright pick sequence ---
    def pick_sequence_upright(self):
        # 0) open
        self.gripper_open()
        # 1) pre-grasp hover
        LOG.info(f"↘️  Pre-grasp ({PREGRASP_X},{PREGRASP_Y},{PREGRASP_Z}), pitch={IK_ALPHA}°")
        self.ik_move((PREGRASP_X, PREGRASP_Y, PREGRASP_Z), IK_TIME_SLOW)
        time.sleep(IK_TIME_SLOW/1000 + 0.1)
        # 2) descend
        LOG.info(f"⬇️  Descend to Z={GRASP_Z:.1f}")
        self.ik_move((PREGRASP_X, PREGRASP_Y, GRASP_Z), IK_TIME_SLOW)
        time.sleep(IK_TIME_SLOW/1000 + 0.1)
        # 3) close
        LOG.info("✊ Close gripper")
        self.gripper_close()
        time.sleep(0.15)
        # 4) lift
        LOG.info(f"⬆️  Lift to Z={LIFT_Z}")
        self.ik_move((PREGRASP_X, PREGRASP_Y, LIFT_Z), IK_TIME_SLOW)
        time.sleep(IK_TIME_SLOW/1000 + 0.1)
        # 5) tiny retract
        self.ik_move((PREGRASP_X - 2, PREGRASP_Y, LIFT_Z), IK_TIME_FAST)
        time.sleep(IK_TIME_FAST/1000 + 0.05)

    # --- Main loop ---
    def run(self):
        LOG.info("🤖 Bruno Bottle Picker (VS upright)")
        LOG.info(f"Images → {paths['images']}")
        LOG.info(f"Bottle keywords → {BOTTLE_KEYWORDS}")
        self.cap = find_cam()
        if not self.cap: return

        self.running = True
        last_frame = None

        try:
            while self.running:
                ok, fr = self.cap.read()
                if ok and fr is not None:
                    last_frame = fr
                else:
                    time.sleep(0.1); continue

                # In approach mode → servoing controller
                if self.bottle_spotted:
                    self.approach_step(last_frame)
                    time.sleep(0.03)
                    continue

                # Roam with ultrasonic
                d = self.ultra.get_distance_cm()
                if d is not None and d <= ULTRA_DANGER_CM:
                    self.ultra.set_rgb(255,0,0)
                    self.car.set_velocity(0,0,0); time.sleep(0.05)
                elif d is not None and d <= ULTRA_CAUTION_CM:
                    self.ultra.set_rgb(255,180,0)
                    left = ((self.frame_idx // 60) % 2 == 0)
                    self.car.set_velocity(0, 90, -0.5 if left else 0.5)
                    time.sleep(TURN_TIME)
                    self.car.set_velocity(0,0,0)
                else:
                    self.ultra.set_rgb(0,255,0)
                    self.car.set_velocity(FORWARD_SPEED, 90, 0)

                # Snapshot + caption
                if last_frame is not None:
                    captext = self.maybe_snapshot_and_caption(last_frame)
                    if captext and looks_like_bottle(captext):
                        LOG.info("🍼 Bottle detected — entering APPROACH (VS)")
                        self.bottle_spotted = True

                self.frame_idx += 1
                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        try: self.car.set_velocity(0,0,0)
        except Exception: pass
        try:
            if self.cap: self.cap.release()
        except Exception: pass
        try: self.ultra.set_rgb(0,0,0)
        except Exception: pass

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    LOG.info("Press Ctrl+C to stop")
    BrunoBottlePickerVS().run()
