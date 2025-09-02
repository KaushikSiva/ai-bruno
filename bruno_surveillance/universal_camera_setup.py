#!/usr/bin/env python3
# coding: utf-8
"""
Universal Camera Setup for Bruno Surveillance
- Detects and configures both built-in and external cameras
- Allows switching between camera types
- Creates configuration files for each setup
- Tests camera functionality
"""

import os
import sys
import time
import json
import subprocess
import threading
import urllib.request
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from urllib.parse import urlparse

import cv2
import requests

# Add Hiwonder SDK path
sys.path.append("/home/pi/MasterPi")

# Import logging from utils if available, otherwise create basic logger
try:
    from utils import LOG
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    LOG.addHandler(handler)

# ===== EXTERNAL CAMERA FUNCTIONS =====

def get_available_external_cameras() -> List[Dict]:
    """Scan for external USB/V4L2 cameras"""
    cams = []
    for i in range(10):
        dp = f"/dev/video{i}"
        if os.path.exists(dp):
            cams.append({
                'path': dp,
                'index': i,
                'accessible': os.access(dp, os.R_OK),
                'type': 'external'
            })
    return cams

def fix_external_camera_permissions():
    """Fix permissions for external cameras"""
    LOG.info('🔧 Checking external camera permissions...')
    try:
        result = subprocess.run(['groups'], capture_output=True, text=True)
        if result.returncode == 0 and 'video' not in result.stdout:
            LOG.warning("⚠️ User not in 'video' group. Run: sudo usermod -a -G video $USER && re-login")
        
        for i in range(10):
            dp = f"/dev/video{i}"
            if os.path.exists(dp) and not os.access(dp, os.R_OK):
                try:
                    subprocess.run(['sudo', 'chmod', '666', dp], timeout=5)
                    LOG.info(f'✅ Fixed permissions for {dp}')
                except Exception as e:
                    LOG.warning(f'⚠️ Could not fix permissions for {dp}: {e}')
    except Exception as e:
        LOG.warning(f'⚠️ Permission check failed: {e}')

def test_external_camera(device_info: Dict) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Test external camera with multiple methods"""
    dp = device_info['path']
    idx = device_info['index']
    LOG.info(f"🎥 Testing external camera {dp} (index {idx})...")

    methods = [
        ('V4L2 with index',  lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2)),
        ('V4L2 with path',   lambda: cv2.VideoCapture(dp,  cv2.CAP_V4L2)),
        ('Default with index', lambda: cv2.VideoCapture(idx)),
        ('Default with path',  lambda: cv2.VideoCapture(dp)),
        ('GStreamer pipeline', lambda: cv2.VideoCapture(f'v4l2src device={dp} ! videoconvert ! appsink', cv2.CAP_GSTREAMER)),
    ]

    for name, func in methods:
        try:
            LOG.info(f'   Trying {name}...')
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
                        LOG.warning(f'   ⚠️ Could not set properties: {e}')
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
    
    return None, "Failed"

# ===== BUILT-IN CAMERA FUNCTIONS =====

def _is_stream_up(url: str, timeout: float = 2.0) -> bool:
    """Check if camera stream is running"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

def _looks_like_local(url: str, default_port: int = 8080) -> bool:
    """Check if URL looks like a local stream"""
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
    """Start the built-in camera stream in background"""
    try:
        # Try to import live_stream_test
        sys.path.append(os.getcwd())
        import live_stream_test
        LOG.info(f"Starting background stream on {host}:{port}")
        t = threading.Thread(
            target=live_stream_test.run_stream, 
            kwargs={"host": host, "port": port}, 
            daemon=True
        )
        t.start()
        return t
    except Exception as e:
        LOG.warning(f"Could not start background stream: {e}")
        # Try alternative method with subprocess
        try:
            LOG.info(f"Trying subprocess method to start stream...")
            subprocess.Popen([
                sys.executable, "-c", 
                f"""
import sys
sys.path.append("/home/pi/MasterPi")
from live_stream_test import CameraStream

if __name__ == "__main__":
    stream = CameraStream(host="{host}", port={port})
    stream.start_stream()
"""
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e2:
            LOG.error(f"Both stream start methods failed: {e}, {e2}")
            return None

def _wait_until(pred, timeout_s: float, interval_s: float = 0.25) -> bool:
    """Wait until predicate becomes true"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if pred():
            return True
        time.sleep(interval_s)
    return False

def test_builtin_camera() -> Tuple[Optional[cv2.VideoCapture], str]:
    """Test built-in camera using stream approach"""
    LOG.info("🎥 Testing built-in camera...")
    
    # Built-in camera URL
    camera_url = os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream")
    stream_port = 8080
    
    # Check if local stream is up, if not start it
    if _looks_like_local(camera_url, stream_port):
        if not _is_stream_up(camera_url):
            LOG.info(f"Local stream {camera_url} not running. Starting it...")
            _start_stream_background(host="0.0.0.0", port=stream_port)
            ok = _wait_until(lambda: _is_stream_up(camera_url), timeout_s=8.0, interval_s=0.5)
            if not ok:
                LOG.warning("Local stream did not come up in time")
    
    # Try to open the camera stream
    methods = [
        ("Built-in stream", lambda: cv2.VideoCapture(camera_url)),
        ("Built-in stream FFMPEG", lambda: cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)),
        ("Device Index 0", lambda: cv2.VideoCapture(0)),
        ("V4L2 Index 0", lambda: cv2.VideoCapture(0, cv2.CAP_V4L2)),
        ("GStreamer CSI", lambda: cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)),
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
                        LOG.warning(f"   ⚠️ Could not set properties: {e}")
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

# ===== CAMERA DETECTION AND SETUP =====

def detect_all_cameras() -> Dict[str, List[Dict]]:
    """Detect all available cameras"""
    LOG.info("🔍 Detecting all available cameras...")
    
    cameras = {
        'builtin': [],
        'external': []
    }
    
    # Test built-in camera
    LOG.info("\n--- Testing Built-in Camera ---")
    builtin_cap, builtin_method = test_builtin_camera()
    if builtin_cap:
        cameras['builtin'].append({
            'type': 'builtin',
            'method': builtin_method,
            'path': 'built-in',
            'working': True
        })
        builtin_cap.release()
        LOG.info("✅ Built-in camera detected and working")
    else:
        LOG.info("❌ Built-in camera not available")
    
    # Test external cameras
    LOG.info("\n--- Testing External Cameras ---")
    fix_external_camera_permissions()
    external_cams = get_available_external_cameras()
    
    if not external_cams:
        LOG.info("❌ No external camera devices found")
    else:
        LOG.info(f"📹 Found {len(external_cams)} external camera device(s)")
        for cam in external_cams:
            if not cam['accessible']:
                LOG.warning(f"⚠️ {cam['path']} not accessible (permissions)")
                continue
            
            cap, method = test_external_camera(cam)
            if cap:
                cam['working'] = True
                cam['method'] = method
                cameras['external'].append(cam)
                cap.release()
                LOG.info(f"✅ External camera {cam['path']} working")
            else:
                LOG.info(f"❌ External camera {cam['path']} not working")
    
    return cameras

def create_camera_config(camera_type: str, camera_data: Dict) -> Dict:
    """Create camera configuration"""
    if camera_type == 'builtin':
        return {
            'type': 'builtin',
            'method': camera_data['method'],
            'camera_url': os.environ.get("BRUNO_CAMERA_URL", "http://127.0.0.1:8080?action=stream"),
            'stream_port': 8080,
            'reconnect_for_photos': True,
            'buffer_flush_count': 10,
            'properties': {
                'buffer_size': 1
            }
        }
    else:  # external
        return {
            'type': 'external',
            'method': camera_data['method'],
            'path': camera_data['path'],
            'index': camera_data['index'],
            'retry_attempts': 3,
            'retry_delay': 2.0,
            'properties': {
                'width': 1280,
                'height': 720,
                'fps': 30,
                'buffer_size': 1
            }
        }

def save_camera_config(config: Dict, config_name: str = "camera_config.json"):
    """Save camera configuration to file"""
    config_path = Path(config_name)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    LOG.info(f"💾 Camera configuration saved to: {config_path}")
    return config_path

def load_camera_config(config_name: str = "camera_config.json") -> Optional[Dict]:
    """Load camera configuration from file"""
    config_path = Path(config_name)
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        LOG.info(f"📂 Camera configuration loaded from: {config_path}")
        return config
    except Exception as e:
        LOG.error(f"❌ Failed to load camera config: {e}")
        return None

def interactive_camera_setup():
    """Interactive camera setup and configuration"""
    print("\n" + "="*60)
    print("🎥 BRUNO UNIVERSAL CAMERA SETUP")
    print("="*60)
    
    # Detect cameras
    cameras = detect_all_cameras()
    
    # Show results
    print(f"\n📋 DETECTION RESULTS:")
    print(f"   Built-in cameras: {len(cameras['builtin'])}")
    print(f"   External cameras: {len(cameras['external'])}")
    
    if not cameras['builtin'] and not cameras['external']:
        print("❌ No working cameras detected!")
        return None
    
    # Let user choose
    options = []
    if cameras['builtin']:
        for i, cam in enumerate(cameras['builtin']):
            options.append(('builtin', cam))
            print(f"   {len(options)}. Built-in Camera ({cam['method']})")
    
    if cameras['external']:
        for cam in cameras['external']:
            options.append(('external', cam))
            print(f"   {len(options)}. External Camera {cam['path']} ({cam['method']})")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nChoose camera (1-{len(options)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                camera_type, camera_data = options[choice_idx]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number or 'q'.")
    
    # Create configuration
    config = create_camera_config(camera_type, camera_data)
    
    # Save configuration
    config_files = {
        'builtin': 'builtin_camera_config.json',
        'external': 'external_camera_config.json'
    }
    
    config_path = save_camera_config(config, config_files[camera_type])
    
    # Also save as default
    save_camera_config(config, 'camera_config.json')
    
    print(f"\n✅ Camera setup complete!")
    print(f"📁 Configuration saved to: {config_path}")
    print(f"📁 Default config: camera_config.json")
    
    # Create environment suggestions
    env_suggestions = []
    if camera_type == 'builtin':
        env_suggestions.extend([
            "CAMERA_TYPE=builtin",
            f"BRUNO_CAMERA_URL={config['camera_url']}",
            f"STREAM_PORT={config['stream_port']}"
        ])
    else:
        env_suggestions.extend([
            "CAMERA_TYPE=external",
            f"EXTERNAL_CAMERA_PATH={config['path']}",
            f"EXTERNAL_CAMERA_INDEX={config['index']}"
        ])
    
    print(f"\n💡 Add these to your .env file:")
    for suggestion in env_suggestions:
        print(f"   {suggestion}")
    
    return config

def test_camera_capture(config: Dict, num_photos: int = 3):
    """Test camera capture with given configuration"""
    LOG.info(f"🧪 Testing camera capture with {config['type']} camera...")
    
    if config['type'] == 'builtin':
        cap, method = test_builtin_camera()
    else:
        # Create fake device info for external camera test
        device_info = {
            'path': config['path'],
            'index': config['index']
        }
        cap, method = test_external_camera(device_info)
    
    if not cap:
        LOG.error("❌ Could not open camera for testing")
        return False
    
    try:
        # Take test photos
        photos_dir = Path("test_photos")
        photos_dir.mkdir(exist_ok=True)
        
        for i in range(num_photos):
            LOG.info(f"📸 Taking test photo {i+1}/{num_photos}...")
            
            # Flush buffer if needed
            if config['type'] == 'builtin':
                for _ in range(5):
                    cap.read()
                    time.sleep(0.05)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                photo_path = photos_dir / f"test_photo_{i+1}_{config['type']}.jpg"
                cv2.imwrite(str(photo_path), frame)
                LOG.info(f"✅ Test photo saved: {photo_path}")
            else:
                LOG.error(f"❌ Failed to capture test photo {i+1}")
                return False
            
            time.sleep(1)  # Wait between photos
        
        LOG.info(f"✅ All {num_photos} test photos captured successfully!")
        return True
        
    finally:
        cap.release()

# ===== MAIN FUNCTIONS =====

def main():
    """Main camera setup function"""
    print("\n🤖 Bruno Universal Camera Setup Tool")
    
    while True:
        print("\nChoose an option:")
        print("1. Detect all cameras")
        print("2. Interactive camera setup")
        print("3. Test current camera config")
        print("4. Switch camera type")
        print("5. Create test photos")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            cameras = detect_all_cameras()
            print(f"\nDetection complete!")
        elif choice == '2':
            config = interactive_camera_setup()
        elif choice == '3':
            config = load_camera_config()
            if config:
                print(f"Current config: {config['type']} camera using {config.get('method', 'unknown method')}")
            else:
                print("No camera configuration found. Run setup first.")
        elif choice == '4':
            # Quick switch between builtin and external
            builtin_config = load_camera_config('builtin_camera_config.json')
            external_config = load_camera_config('external_camera_config.json')
            
            if builtin_config and external_config:
                print("Available configs:")
                print("1. Built-in camera")
                print("2. External camera")
                switch_choice = input("Switch to (1/2): ").strip()
                
                if switch_choice == '1':
                    save_camera_config(builtin_config, 'camera_config.json')
                    print("✅ Switched to built-in camera")
                elif switch_choice == '2':
                    save_camera_config(external_config, 'camera_config.json')
                    print("✅ Switched to external camera")
            else:
                print("❌ Not all camera configs available. Run setup first.")
        elif choice == '5':
            config = load_camera_config()
            if config:
                test_camera_capture(config)
            else:
                print("No camera configuration found. Run setup first.")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()