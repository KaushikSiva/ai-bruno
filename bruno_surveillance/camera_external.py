import os, sys, time, subprocess
from typing import Optional, Dict, Tuple, List

import cv2
from utils import LOG

# ------------------------------------------------------------------------------
# Arm neutralizer: prevent the arm from moving to default pose when this module loads
# Enabled by default. Set EXTERNAL_CAM_NO_ARM=0 to skip.
# ------------------------------------------------------------------------------
if os.environ.get("EXTERNAL_CAM_NO_ARM", "1") == "1":
    try:
        sys.path.append("/home/pi/MasterPi")
        # Import only Board (not ArmIK) to avoid any IK motions
        from common.ros_robot_controller_sdk import Board  # type: ignore
        _board = Board()
        # Typical arm/gripper channels (adjust if needed)
        for ch in (1, 2, 3, 4, 5, 6):
            try:
                _board.pwm_servo_enable(ch, False)
            except Exception:
                pass
        LOG.info("🛡️  camera_external: arm PWM disabled on channels 1–6. "
                 "Set EXTERNAL_CAM_NO_ARM=0 to skip.")
    except Exception as _e:
        LOG.warning(f"camera_external: arm neutralizer skipped: {_e}")
# ------------------------------------------------------------------------------

def get_available_cameras(limit: int = 10) -> List[Dict]:
    cams = []
    for i in range(limit):
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

def _set_fourcc_and_props(cap: cv2.VideoCapture) -> None:
    """Force sane pixel format & props to avoid warped/bent frames."""
    for fourcc in ("MJPG", "YUYV"):
        try:
            ok = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            LOG.info(f"   🎞️  Set FOURCC={fourcc} ok={ok}")
        except Exception:
            pass
    # Best-effort property negotiation (ignore failures)
    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

def _warmup_and_validate(cap: cv2.VideoCapture, reads: int = 6) -> Optional[Tuple[int, int]]:
    # discard few frames to stabilize auto exposure/focus
    for _ in range(4):
        cap.read(); time.sleep(0.02)
    for _ in range(reads):
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                return (w, h)
        time.sleep(0.03)
    return None

def open_external_camera_robust(device_info: Dict) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    dp = device_info["path"]; idx = device_info["index"]
    LOG.info(f"🎥 Attempting to open {dp} (index {idx})…")

    methods = [
        ("V4L2 with index",    lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ("V4L2 with path",     lambda: cv2.VideoCapture(dp,  cv2.CAP_V4L2)),
        ("Default with index", lambda: cv2.VideoCapture(idx)),
        ("Default with path",  lambda: cv2.VideoCapture(dp)),
        # Explicit MJPEG decode pipeline to avoid “bending”:
        ("GStreamer MJPEG",
         lambda: cv2.VideoCapture(
             f"v4l2src device={dp} ! image/jpeg,framerate=30/1 ! "
             f"jpegparse ! jpegdec ! videoconvert ! appsink",
             cv2.CAP_GSTREAMER)),
        # If your camera outputs YUY2 only, try this instead:
        # ("GStreamer YUY2",
        #  lambda: cv2.VideoCapture(
        #      f"v4l2src device={dp} ! video/x-raw,format=YUY2,framerate=30/1 ! "
        #      f"videoconvert ! appsink",
        #      cv2.CAP_GSTREAMER)),
    ]

    for name, fn in methods:
        try:
            LOG.info(f"   Trying {name}…")
            cap = fn()
            if not (cap and cap.isOpened()):
                if cap: cap.release()
                continue

            _set_fourcc_and_props(cap)
            wh = _warmup_and_validate(cap)
            if wh:
                LOG.info(f"   ✅ {name} SUCCESS! Resolution: {wh[0]}x{wh[1]}")
                return cap, name

            LOG.warning(f"   ❌ {name} opened but invalid/warped frames; releasing…")
            cap.release()
        except Exception as e:
            LOG.warning(f"   ❌ {name} error: {e}")
    return None, None

def find_working_external_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[Dict]]:
    LOG.info("🔍 Scanning for external cameras…")
    fix_camera_permissions()
    cams = get_available_cameras()
    if not cams:
        LOG.error("❌ No camera devices found at /dev/video*"); return None, None
    LOG.info(f"📹 Found {len(cams)} camera device(s)")
    for c in sorted(cams, key=lambda x: x["index"]):
        cap, method = open_external_camera_robust(c)
        if cap:
            LOG.info(f"✅ Successfully opened {c['path']} using {method}")
            return cap, c
    LOG.error("❌ No working external cameras found"); return None, None

class ExternalCamera:
    def __init__(self, retry_attempts: int = 3, retry_delay: float = 2.0):
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.cap = None
        self.info = None

    def open(self) -> bool:
        for attempt in range(self.retry_attempts):
            LOG.info(f"📹 Camera connection attempt {attempt+1}…")
            self.cap, self.info = find_working_external_camera()
            if self.cap:
                LOG.info(f"✅ External camera connected: {self.info['path']}")
                return True
            if attempt < self.retry_attempts - 1:
                LOG.info(f"⏳ Retrying in {self.retry_delay} seconds…")
                time.sleep(self.retry_delay)
        LOG.error("❌ Failed to connect external camera"); return False

    def is_open(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def read(self):
        if not self.is_open():
            return False, None
        ok, frame = self.cap.read()
        return (ok and frame is not None), (frame if ok else None)

    def get_fresh_frame(self, max_attempts: int = 3, settle_reads: int = 3):
        if not self.is_open():
            return None
        try:
            for _ in range(settle_reads):
                self.cap.read(); time.sleep(0.02)
            for _ in range(max_attempts):
                ok, f = self.cap.read()
                if ok and f is not None:
                    return f
                time.sleep(0.03)
        except Exception:
            return None
        return None

    def reopen(self):
        try:
            self.release()
        except Exception:
            pass
        time.sleep(1.0)
        self.open()

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.info = None
