import os, time, subprocess
from typing import Optional, Dict, Tuple, List

import cv2
from utils import LOG

def get_available_cameras(limit: int = 10) -> List[Dict]:
    cams = []
    for i in range(limit):
        dp = f'/dev/video{i}'
        if os.path.exists(dp):
            cams.append({'path': dp, 'index': i, 'accessible': os.access(dp, os.R_OK)})
    return cams

def fix_camera_permissions():
    LOG.info('🔧 Checking camera permissions...')
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if result.returncode == 0 and 'video' not in result.stdout:
            LOG.warning("⚠️  User not in 'video' group. Run: sudo usermod -a -G video $USER && re-login")
        for i in range(5):
            dp = f'/dev/video{i}'
            if os.path.exists(dp) and not os.access(dp, os.R_OK):
                try:
                    subprocess.run(['sudo', 'chmod', '666', dp], timeout=5)
                    LOG.info(f'✅ Fixed permissions for {dp}')
                except Exception as e:
                    LOG.warning(f'⚠️  Could not fix permissions for {dp}: {e}')
    except Exception as e:
        LOG.warning(f'⚠️  Permission check failed: {e}')

def open_external_camera_robust(device_info: Dict) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    dp = device_info['path']; idx = device_info['index']
    LOG.info(f'🎥 Attempting to open {dp} (index {idx})…')
    methods = [
        ('V4L2 with index',  lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ('V4L2 with path',   lambda: cv2.VideoCapture(dp,  cv2.CAP_V4L2)),
        ('Default with index', lambda: cv2.VideoCapture(idx)),
        ('Default with path',  lambda: cv2.VideoCapture(dp)),
        ('GStreamer pipeline', lambda: cv2.VideoCapture(f'v4l2src device={dp} ! videoconvert ! appsink', cv2.CAP_GSTREAMER)),
    ]
    for name, fn in methods:
        try:
            LOG.info(f'   Trying {name}…')
            cap = fn()
            if cap and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    LOG.info(f'   ✅ {name} SUCCESS! Resolution: {w}x{h}')
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception: pass
                    return cap, name
                cap.release()
            else:
                if cap: cap.release()
        except Exception as e:
            LOG.warning(f'   ❌ {name} error: {e}')
    return None, None

def find_working_external_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[Dict]]:
    LOG.info('🔍 Scanning for external cameras…')
    fix_camera_permissions()
    cams = get_available_cameras()
    if not cams:
        LOG.error('❌ No camera devices found at /dev/video*'); return None, None
    LOG.info(f'📹 Found {len(cams)} camera device(s)')
    for c in sorted(cams, key=lambda x: x['index']):
        cap, method = open_external_camera_robust(c)
        if cap:
            LOG.info(f'✅ Successfully opened {c['path']} using {method}')
            return cap, c
    LOG.error('❌ No working external cameras found'); return None, None

class ExternalCamera:
    def __init__(self, retry_attempts: int = 3, retry_delay: float = 2.0):
        self.retry_attempts = retry_attempts; self.retry_delay = retry_delay
        self.cap = None; self.info = None
    def open(self) -> bool:
        for attempt in range(self.retry_attempts):
            LOG.info(f'📹 Camera connection attempt {attempt+1}…')
            self.cap, self.info = find_working_external_camera()
            if self.cap:
                LOG.info(f"✅ External camera connected: {self.info['path']}"); return True
            if attempt < self.retry_attempts - 1:
                LOG.info(f'⏳ Retrying in {self.retry_delay} seconds…'); time.sleep(self.retry_delay)
        LOG.error('❌ Failed to connect external camera'); return False
    def is_open(self) -> bool:
        return bool(self.cap and self.cap.isOpened())
    def read(self):
        if not self.is_open(): return False, None
        ok, frame = self.cap.read(); return (ok and frame is not None), (frame if ok else None)
    def get_fresh_frame(self, max_attempts: int = 3, settle_reads: int = 3):
        if not self.is_open(): return None
        try:
            for _ in range(settle_reads): self.cap.read(); time.sleep(0.02)
            for _ in range(max_attempts):
                ok, f = self.cap.read()
                if ok and f is not None: return f
                time.sleep(0.03)
        except Exception: return None
        return None
    def reopen(self):
        try: self.release()
        except Exception: pass
        time.sleep(1.0); self.open()
    def release(self):
        try:
            if self.cap: self.cap.release()
        except Exception: pass
        self.cap = None; self.info = None
