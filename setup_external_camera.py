#!/usr/bin/env python3
"""
Bruno External Camera Setup Script
Complete setup to configure Bruno to use your external USB camera
"""

import cv2
import json
import os
import sys

def detect_cameras():
    """Detect available cameras using multiple backends"""
    print("Bruno External Camera Setup")
    print("=" * 40)
    
    working_cameras = []
    
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name}...")
        
        for device_id in range(5):
            try:
                print(f"  Camera {device_id}...", end=" ")
                cap = cv2.VideoCapture(device_id, backend_id)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"[OK] {width}x{height}")
                        
                        working_cameras.append({
                            'device_id': device_id,
                            'width': width,
                            'height': height,
                            'backend': backend_name
                        })
                        cap.release()
                        break  # Found working camera, stop testing this backend
                    else:
                        print("[FAIL] No frame")
                        cap.release()
                else:
                    print("[FAIL] Cannot open")
                    
            except Exception as e:
                print(f"[ERROR] {e}")
                
        if working_cameras:
            break  # Found working cameras, stop testing other backends
    
    return working_cameras

def create_bruno_config(camera_device_id, width=640, height=480):
    """Create or update Bruno configuration for external camera"""
    config = {
        "camera": {
            "device_id": camera_device_id,
            "fallback_device_id": 0,
            "width": width,
            "height": height,
            "fps": 30,
            "flip_horizontal": False,
            "max_camera_failures": 10,
            "max_recovery_attempts": 3
        },
        "detection": {
            "color_detection": {
                "clear_plastic": {
                    "lower_hsv": [0, 0, 150],
                    "upper_hsv": [180, 50, 255]
                },
                "blue_plastic": {
                    "lower_hsv": [100, 150, 50],
                    "upper_hsv": [130, 255, 255]
                },
                "green_plastic": {
                    "lower_hsv": [40, 100, 50],
                    "upper_hsv": [80, 255, 255]
                }
            },
            "size_filter": {
                "min_area": 1000,
                "max_area": 50000,
                "min_aspect_ratio": 1.2,
                "max_aspect_ratio": 4.5
            },
            "morphology": {
                "kernel_size": 3,
                "iterations": 2
            },
            "confidence_threshold": 0.6,
            "detection_cooldown": 1.0,
            "auto_approach": True
        },
        "distance_estimation": {
            "focal_length": 500,
            "real_bottle_height": 20,
            "camera_height": 25,
            "camera_tilt": 15,
            "stop_distance": 30,
            "approach_distance": 100,
            "max_detection_distance": 200
        },
        "movement_control": {
            "enabled": True,
            "max_speed": 40,
            "min_speed": 15,
            "turn_multiplier": 0.8,
            "movement_timeout": 0.5,
            "approach_enabled": True,
            "safety_enabled": True,
            "smooth_acceleration": True
        },
        "behavior": {
            "celebration_on_detection": True,
            "approach_bottles_automatically": True,
            "stop_on_reach": True,
            "max_detection_responses": 5
        }
    }
    
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # Backup existing config if it exists
    config_file = "config/bruno_config.json"
    if os.path.exists(config_file):
        backup_file = "config/bruno_config_backup.json"
        with open(config_file, 'r') as f:
            backup_config = json.load(f)
        with open(backup_file, 'w') as f:
            json.dump(backup_config, f, indent=4)
        print(f"[OK] Backed up existing config to {backup_file}")
    
    # Write new config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"[OK] Created Bruno configuration: {config_file}")
    return config_file

def create_camera_switch_script():
    """Create a script to easily switch between cameras"""
    script_content = '''#!/usr/bin/env python3
"""
Camera Switch Script for Bruno
Usage: python camera_switch.py [device_id]
"""

import json
import sys
import os

def update_camera_device(device_id):
    """Update camera device in Bruno config"""
    config_file = "config/bruno_config.json"
    
    if not os.path.exists(config_file):
        print("[ERROR] Bruno config file not found!")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        config['camera']['device_id'] = device_id
        config['camera']['fallback_device_id'] = device_id
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"[OK] Switched to camera device {device_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update config: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python camera_switch.py [device_id]")
        print("Example: python camera_switch.py 0")
        print("         python camera_switch.py 1")
        sys.exit(1)
    
    try:
        device_id = int(sys.argv[1])
        update_camera_device(device_id)
    except ValueError:
        print("[ERROR] Device ID must be a number")
        sys.exit(1)
'''
    
    with open('camera_switch.py', 'w') as f:
        f.write(script_content)
    
    print("[OK] Created camera_switch.py")

def test_camera_capture(device_id):
    """Test capturing from the specified camera"""
    print(f"\nTesting camera {device_id}...")
    
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {device_id}")
        return False
    
    # Capture a test image
    ret, frame = cap.read()
    if ret:
        filename = f"camera_{device_id}_test.jpg"
        cv2.imwrite(filename, frame)
        print(f"[OK] Test image saved: {filename}")
        
        # Show camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera properties: {width}x{height} @ {fps:.1f}fps")
        
        cap.release()
        return True
    else:
        print(f"[ERROR] Cannot capture from camera {device_id}")
        cap.release()
        return False

def main():
    print("Setting up Bruno to use external camera...")
    print("Make sure your external camera is connected!")
    print()
    
    # Detect cameras
    cameras = detect_cameras()
    
    if not cameras:
        print("\n[INFO] No cameras detected right now.")
        print("To use your external camera when it's connected:")
        print("1. Connect your external USB camera")
        print("2. Run this script again")
        print("3. Or manually edit config/bruno_config.json")
        
        # Create default config for device 1 (common for external cameras)
        create_bruno_config(1)
        create_camera_switch_script()
        
        print("\n[INFO] Created default configuration for camera device 1")
        print("If your camera uses a different device ID, use:")
        print("  python camera_switch.py [device_id]")
        return
    
    print(f"\nFound {len(cameras)} working cameras:")
    for i, cam in enumerate(cameras):
        print(f"  {i+1}. Device {cam['device_id']}: {cam['width']}x{cam['height']} ({cam['backend']})")
    
    # If multiple cameras, let user choose
    if len(cameras) > 1:
        while True:
            try:
                choice = input(f"\nSelect camera (1-{len(cameras)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(cameras):
                    selected_camera = cameras[choice_idx]
                    break
                else:
                    print(f"Please enter 1-{len(cameras)}")
            except ValueError:
                print("Please enter a number")
    else:
        selected_camera = cameras[0]
    
    # Test the selected camera
    if not test_camera_capture(selected_camera['device_id']):
        print("[ERROR] Camera test failed!")
        return
    
    # Create configuration
    config_file = create_bruno_config(
        selected_camera['device_id'],
        selected_camera['width'],
        selected_camera['height']
    )
    
    # Create switch script
    create_camera_switch_script()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print(f"[OK] Bruno configured for camera device {selected_camera['device_id']}")
    print(f"[OK] Configuration saved to {config_file}")
    print(f"[OK] Camera switch script created: camera_switch.py")
    print("\nYour Bruno robot is now ready to use the external camera!")
    print("\nTo switch cameras later:")
    print("  python camera_switch.py 0  # Use device 0")
    print("  python camera_switch.py 1  # Use device 1")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Setup cancelled.")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")