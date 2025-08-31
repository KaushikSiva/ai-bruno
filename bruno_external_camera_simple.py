#!/usr/bin/env python3
"""
Bruno External Camera Script
Configures Bruno to use an external USB camera instead of the built-in camera.
"""

import cv2
import json
import os
import sys
import time

class BrunoExternalCameraSetup:
    def __init__(self):
        self.config_file = "config/bruno_config.json"
        self.backup_config_file = "config/bruno_config_backup.json"
        
    def detect_cameras(self):
        """Detect all available cameras"""
        print("Scanning for available cameras...")
        cameras = []
        
        # Test USB cameras (device IDs 0-5)
        for device_id in range(6):
            try:
                print(f"Testing camera device {device_id}...", end=" ")
                cap = cv2.VideoCapture(device_id)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"[OK] Working! Resolution: {width}x{height}")
                        cameras.append({
                            'type': 'USB',
                            'device_id': device_id,
                            'width': width,
                            'height': height
                        })
                    else:
                        print("[FAIL] Opens but no frame")
                    cap.release()
                else:
                    print("[FAIL] Cannot open")
            except Exception as e:
                print(f"[ERROR] {e}")
                
        return cameras
    
    def backup_config(self):
        """Backup current configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                with open(self.backup_config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"[OK] Configuration backed up to {self.backup_config_file}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to backup config: {e}")
                return False
        else:
            print("[INFO] No existing config file found")
            return True
    
    def update_config_for_external_camera(self, camera_info):
        """Update Bruno config to use external camera"""
        # Load existing config or create new one
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "camera": {},
                "detection": {
                    "color_detection": {
                        "clear_plastic": {
                            "lower_hsv": [0, 0, 150],
                            "upper_hsv": [180, 50, 255]
                        }
                    }
                }
            }
        
        # Update camera configuration
        config['camera'] = {
            'device_id': camera_info['device_id'],
            'fallback_device_id': camera_info.get('device_id', 0),
            'width': camera_info.get('width', 640),
            'height': camera_info.get('height', 480),
            'fps': 30,
            'flip_horizontal': False,
            'max_camera_failures': 10,
            'max_recovery_attempts': 3
        }
        
        # Save updated config
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"[OK] Updated {self.config_file} for external camera")
        return True
    
    def test_external_camera(self, camera_info):
        """Test external camera with a quick capture"""
        print(f"Testing external camera (device {camera_info['device_id']})...")
        
        cap = cv2.VideoCapture(camera_info['device_id'])
        
        if not cap.isOpened():
            print("[ERROR] Failed to open camera")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_info.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_info.get('height', 480))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Capture test frames
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
            time.sleep(0.1)
        
        cap.release()
        
        if success_count >= 3:
            print(f"[OK] External camera working! ({success_count}/5 frames captured)")
            
            # Save a test image
            cap = cv2.VideoCapture(camera_info['device_id'])
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('external_camera_test.jpg', frame)
                print("[OK] Test image saved: external_camera_test.jpg")
            cap.release()
            
            return True
        else:
            print(f"[ERROR] External camera unstable ({success_count}/5 frames)")
            return False
    
    def run(self):
        """Main setup process"""
        print("=" * 50)
        print("Bruno External Camera Setup")
        print("=" * 50)
        
        # Detect cameras
        cameras = self.detect_cameras()
        
        if not cameras:
            print("[ERROR] No working cameras found!")
            print("Make sure your external USB camera is connected.")
            return False
        
        print(f"\nFound {len(cameras)} working cameras:")
        external_cameras = [cam for cam in cameras if cam['type'] == 'USB']
        
        for i, cam in enumerate(cameras):
            print(f"  {i+1}. {cam['type']} Camera - Device {cam['device_id']} ({cam['width']}x{cam['height']})")
        
        if not external_cameras:
            print("\n[ERROR] No external USB cameras found!")
            print("Make sure your external camera is connected and working.")
            return False
        
        # Use the first external camera (likely your new camera)
        # If device 0 is built-in, external camera is usually device 1
        external_camera = None
        for cam in external_cameras:
            if cam['device_id'] > 0:  # Prefer non-zero device IDs for external cameras
                external_camera = cam
                break
        
        if not external_camera:
            external_camera = external_cameras[0]  # Fallback to first found
        
        print(f"\n[INFO] Selected external camera: Device {external_camera['device_id']}")
        
        # Backup existing config
        print("\nBacking up configuration...")
        if not self.backup_config():
            response = input("Continue without backup? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Test external camera
        print("\nTesting external camera...")
        if not self.test_external_camera(external_camera):
            print("[ERROR] External camera test failed!")
            return False
        
        # Update config
        print("\nUpdating configuration for external camera...")
        self.update_config_for_external_camera(external_camera)
        
        print("\n" + "=" * 50)
        print("SETUP COMPLETE!")
        print("=" * 50)
        print(f"[OK] Bruno is now configured to use external camera (device {external_camera['device_id']})")
        print(f"[OK] Configuration backup saved to {self.backup_config_file}")
        print(f"[OK] Test image saved: external_camera_test.jpg")
        print("\n[INFO] You can now run your Bruno scripts with the external camera!")
        
        return True

def main():
    try:
        setup = BrunoExternalCameraSetup()
        success = setup.run()
        
        if not success:
            print("\n[ERROR] Setup failed. Check the issues above and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()