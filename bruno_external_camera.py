#!/usr/bin/env python3
"""
Bruno External Camera Script
Configures Bruno to use an external USB camera instead of the built-in camera.
Detects available cameras and switches to the external one.
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
        
        # Test USB cameras (device IDs 0-10)
        for device_id in range(11):
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
                            'height': height,
                            'source': device_id
                        })
                    else:
                        print("[FAIL] Opens but no frame")
                    cap.release()
                else:
                    print("[FAIL] Cannot open")
            except Exception as e:
                print(f"[ERROR] Error: {e}")
                
        # Test built-in camera stream
        builtin_sources = [
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream'
        ]
        
        for source in builtin_sources:
            try:
                print(f"Testing built-in camera: {source}...", end=" ")
                cap = cv2.VideoCapture(source)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"✓ Working! Resolution: {width}x{height}")
                        cameras.append({
                            'type': 'Built-in',
                            'device_id': source,
                            'width': width,
                            'height': height,
                            'source': source
                        })
                    else:
                        print("✗ Opens but no frame")
                    cap.release()
                else:
                    print("✗ Cannot open")
            except Exception as e:
                print(f"✗ Error: {e}")
                
        return cameras
    
    def backup_config(self):
        """Backup current configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                with open(self.backup_config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"✓ Configuration backed up to {self.backup_config_file}")
                return True
            except Exception as e:
                print(f"✗ Failed to backup config: {e}")
                return False
        else:
            print("ℹ️  No existing config file found")
            return True
    
    def update_config_for_external_camera(self, camera_info):
        """Update Bruno config to use external camera"""
        # Load existing config or create new one
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update camera configuration
        config['camera'] = {
            'device_id': camera_info['device_id'],
            'fallback_device_id': camera_info.get('device_id', 0),
            'width': camera_info.get('width', 640),
            'height': camera_info.get('height', 480),
            'fps': 30,
            'flip_horizontal': False,  # May need adjustment for external camera
            'max_camera_failures': 10,
            'max_recovery_attempts': 3
        }
        
        # Save updated config
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✓ Updated {self.config_file} for external camera")
        return True
    
    def test_camera_with_config(self, camera_info):
        """Test camera with a quick capture"""
        print(f"🧪 Testing external camera (device {camera_info['device_id']})...")
        
        cap = cv2.VideoCapture(camera_info['device_id'])
        
        if not cap.isOpened():
            print("✗ Failed to open camera")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_info.get('width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_info.get('height', 480))
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Capture a few test frames
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
            time.sleep(0.1)
        
        cap.release()
        
        if success_count >= 3:
            print(f"✓ External camera working! ({success_count}/5 frames captured)")
            
            # Save a test image
            cap = cv2.VideoCapture(camera_info['device_id'])
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('external_camera_test.jpg', frame)
                print("✓ Test image saved: external_camera_test.jpg")
            cap.release()
            
            return True
        else:
            print(f"✗ External camera unstable ({success_count}/5 frames)")
            return False
    
    def create_camera_switch_script(self, external_camera, builtin_camera=None):
        """Create a script to easily switch between cameras"""
        script_content = f'''#!/usr/bin/env python3
"""
Camera Switch Script for Bruno
Quickly switch between external and built-in cameras
"""

import json
import shutil
import os

EXTERNAL_CAMERA_CONFIG = {{
    "camera": {{
        "device_id": {external_camera['device_id']},
        "fallback_device_id": {external_camera.get('device_id', 0)},
        "width": {external_camera.get('width', 640)},
        "height": {external_camera.get('height', 480)},
        "fps": 30,
        "flip_horizontal": false,
        "max_camera_failures": 10,
        "max_recovery_attempts": 3
    }}
}}

BUILTIN_CAMERA_CONFIG = {{
    "camera": {{
        "device_id": "http://127.0.0.1:8080?action=stream",
        "fallback_device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": true,
        "max_camera_failures": 10,
        "max_recovery_attempts": 3
    }}
}}

def switch_to_external():
    """Switch to external USB camera"""
    print("🔄 Switching to external camera...")
    update_camera_config(EXTERNAL_CAMERA_CONFIG)
    print("✓ Switched to external camera (device {external_camera['device_id']})")

def switch_to_builtin():
    """Switch to built-in camera"""
    print("🔄 Switching to built-in camera...")
    update_camera_config(BUILTIN_CAMERA_CONFIG)
    print("✓ Switched to built-in camera")

def update_camera_config(camera_config):
    """Update camera configuration"""
    config_file = "config/bruno_config.json"
    
    # Load existing config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {{}}
    
    # Update camera section
    config.update(camera_config)
    
    # Save config
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python camera_switch.py external  # Switch to external camera")
        print("  python camera_switch.py builtin   # Switch to built-in camera")
        sys.exit(1)
    
    choice = sys.argv[1].lower()
    
    if choice == "external":
        switch_to_external()
    elif choice == "builtin":
        switch_to_builtin()
    else:
        print("Invalid option. Use 'external' or 'builtin'")
        sys.exit(1)
'''
        
        with open('camera_switch.py', 'w') as f:
            f.write(script_content)
        
        print("✓ Created camera_switch.py script")
        print("Usage:")
        print("  python camera_switch.py external  # Switch to external camera") 
        print("  python camera_switch.py builtin   # Switch to built-in camera")
    
    def run(self):
        """Main setup process"""
        print("Bruno External Camera Setup")
        print("=" * 50)
        
        # Detect cameras
        cameras = self.detect_cameras()
        
        if not cameras:
            print("❌ No working cameras found!")
            return False
        
        print(f"\n📷 Found {len(cameras)} working cameras:")
        external_cameras = []
        builtin_camera = None
        
        for i, cam in enumerate(cameras):
            print(f"  {i+1}. {cam['type']} Camera - Device {cam['device_id']} ({cam['width']}x{cam['height']})")
            if cam['type'] == 'USB':
                external_cameras.append(cam)
            else:
                builtin_camera = cam
        
        if not external_cameras:
            print("\n❌ No external USB cameras found!")
            print("Make sure your external camera is connected and working.")
            return False
        
        # Select external camera (use the first one if multiple)
        external_camera = external_cameras[0]
        if len(external_cameras) > 1:
            print(f"\n🔍 Multiple external cameras found. Using device {external_camera['device_id']}.")
        
        # Backup existing config
        print(f"\n💾 Backing up configuration...")
        if not self.backup_config():
            response = input("Continue without backup? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Test external camera
        if not self.test_camera_with_config(external_camera):
            print("❌ External camera test failed!")
            return False
        
        # Update config
        print(f"\n⚙️  Updating configuration for external camera...")
        self.update_config_for_external_camera(external_camera)
        
        # Create switch script
        print(f"\n🔧 Creating camera switch script...")
        self.create_camera_switch_script(external_camera, builtin_camera)
        
        print("\n" + "=" * 50)
        print("✅ SETUP COMPLETE!")
        print("=" * 50)
        print(f"✓ Bruno is now configured to use external camera (device {external_camera['device_id']})")
        print(f"✓ Configuration backup saved to {self.backup_config_file}")
        print(f"✓ Test image saved: external_camera_test.jpg")
        print(f"✓ Camera switch script created: camera_switch.py")
        print("\n🚀 You can now run your Bruno scripts with the external camera!")
        print("\n📝 To switch cameras later:")
        print("  python camera_switch.py external  # Use external camera")
        print("  python camera_switch.py builtin   # Use built-in camera")
        
        return True

def main():
    try:
        setup = BrunoExternalCameraSetup()
        success = setup.run()
        
        if not success:
            print("\n❌ Setup failed. Check the issues above and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()