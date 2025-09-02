import os, time, subprocess
from typing import Optional, Dict, Tuple, List

import cv2
from utils import LOG

def get_available_cameras() -> List[Dict]:
    cams = []
    for i in range(10):
        dp = f"/dev/video{i}"
        if os.path.exists(dp):
            cams.append({
                'path': dp,
                'index': i,
                'accessible': os.access(dp, os.R_OK)
            })
    return cams

def fix_camera_permissions():
    LOG.info('🔧 Checking camera permissions...')
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if result.returncode == 0 and 'video' not in result.stdout:
            LOG.warning("⚠️  User not in 'video' group. Run: sudo usermod -a -G video $USER && re-login")
        for i in range(5):
            dp = f"/dev/video{i}"
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
    LOG.info(f"🎥 Attempting to open {dp} (index {idx})…")

    methods = [
        ('V4L2 with index',  lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ('V4L2 with path',   lambda: cv2.VideoCapture(dp,  cv2.CAP_V4L2)),
        ('Default with index', lambda: cv2.VideoCapture(idx)),
        ('Default with path',  lambda: cv2.VideoCapture(dp)),
        ('GStreamer pipeline', lambda: cv2.VideoCapture(f'v4l2src device={dp} ! videoconvert ! appsink', cv2.CAP_GSTREAMER)),
    ]

    for name, func in methods:
        try:
            LOG.info(f'   Trying {name}…')
            cap = func()
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    LOG.info(f'   ✅ {name} SUCCESS! Resolution: {w}x{h}')
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        LOG.info('   📐 Camera properties set')
                    except Exception as e:
                        LOG.warning(f'   ⚠️  Could not set properties: {e}')
                    return cap, name
                else:
                    LOG.warning(f'   ❌ {name} opened but cannot read frames')
                    cap.release()
            else:
                LOG.warning(f'   ❌ {name} failed to open')
                if cap:
                    cap.release()
        except Exception as e:
            LOG.warning(f'   ❌ {name} error: {e}')
    return None, None

def find_working_external_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[Dict]]:
    LOG.info('🔍 Scanning for external cameras…')
    fix_camera_permissions()
    cams = get_available_cameras()
    if not cams:
        LOG.error('❌ No camera devices found at /dev/video*')
        return None, None
    LOG.info(f"📹 Found {len(cams)} camera device(s)")
    for c in cams:
        if not c['accessible']:
            LOG.warning(f"⚠️  {c['path']} not accessible (permissions issue)")
    for c in sorted(cams, key=lambda x: x['index']):
        cap, method = open_external_camera_robust(c)
        if cap:
            LOG.info(f"✅ Successfully opened {c['path']} using {method}")
            return cap, c
    LOG.error('❌ No working external cameras found')
    return None, None
