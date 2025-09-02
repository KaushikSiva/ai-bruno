#!/usr/bin/env python3
# coding: utf-8
"""
External Camera Fix for Bruno Surveillance
- Fixes V4L2 camera access issues
- Provides camera diagnostic and setup
- Creates working camera configuration
"""

import os
import sys
import time
import subprocess
import cv2

def check_camera_devices():
    """Check what camera devices are available on the system"""
    print("🔍 Checking available camera devices...")
    
    # Method 1: Check /dev/video* devices
    video_devices = []
    for i in range(10):  # Check video0 to video9
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    print(f"📹 Found video devices: {video_devices}")
    
    # Method 2: Use v4l2-ctl if available
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("📋 v4l2-ctl device list:")
            print(result.stdout)
        else:
            print("⚠️ v4l2-ctl not available or failed")
    except Exception as e:
        print(f"⚠️ Could not run v4l2-ctl: {e}")
    
    # Method 3: Use lsusb to check USB devices
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("🔌 USB devices (looking for cameras):")
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'video', 'imaging']):
                    print(f"   {line}")
    except Exception as e:
        print(f"⚠️ Could not run lsusb: {e}")
    
    return video_devices

def test_camera_access(device_path, index=None):
    """Test different ways to access a camera device"""
    print(f"\n🧪 Testing camera access: {device_path}")
    
    methods = [
        ("Direct path", device_path),
        ("Index only", index if index is not None else 0),
        ("V4L2 backend + path", (device_path, cv2.CAP_V4L2)),
        ("V4L2 backend + index", (index if index is not None else 0, cv2.CAP_V4L2)),
        ("FFMPEG backend + path", (device_path, cv2.CAP_FFMPEG)),
        ("GStreamer backend", (index if index is not None else 0, cv2.CAP_GSTREAMER)),
    ]
    
    working_methods = []
    
    for method_name, method_args in methods:
        print(f"   Trying {method_name}...")
        try:
            if isinstance(method_args, tuple):
                cap = cv2.VideoCapture(*method_args)
            else:
                cap = cv2.VideoCapture(method_args)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"   ✅ {method_name} WORKS! Resolution: {width}x{height}")
                    working_methods.append((method_name, method_args))
                else:
                    print(f"   ❌ {method_name} opened but can't read frames")
            else:
                print(f"   ❌ {method_name} failed to open")
            
            cap.release()
            
        except Exception as e:
            print(f"   ❌ {method_name} error: {e}")
    
    return working_methods

def fix_camera_permissions():
    """Fix common camera permission issues"""
    print("\n🔧 Checking camera permissions...")
    
    # Check if user is in video group
    try:
        import pwd
        import grp
        
        username = pwd.getpwuid(os.getuid()).pw_name
        video_group = grp.getgrnam('video')
        video_users = video_group.gr_mem
        
        if username in video_users:
            print(f"✅ User '{username}' is in video group")
        else:
            print(f"❌ User '{username}' is NOT in video group")
            print("💡 Fix with: sudo usermod -a -G video $USER")
            print("   Then logout and login again")
            
    except Exception as e:
        print(f"⚠️ Could not check video group: {e}")
    
    # Check device permissions
    for i in range(5):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            try:
                stat_info = os.stat(device_path)
                permissions = oct(stat_info.st_mode)[-3:]
                print(f"📹 {device_path} permissions: {permissions}")
                
                # Check if readable
                if os.access(device_path, os.R_OK):
                    print(f"   ✅ {device_path} is readable")
                else:
                    print(f"   ❌ {device_path} is NOT readable")
                    print(f"   💡 Fix with: sudo chmod 666 {device_path}")
                    
            except Exception as e:
                print(f"   ⚠️ Could not check {device_path}: {e}")

def create_camera_config(working_methods, device_path):
    """Create a camera configuration based on working methods"""
    print(f"\n📝 Creating camera configuration...")
    
    if not working_methods:
        print("❌ No working camera methods found!")
        return None
    
    # Prefer methods in this order
    preferred_order = [
        "V4L2 backend + index",
        "V4L2 backend + path", 
        "Direct path",
        "Index only",
        "FFMPEG backend + path",
        "GStreamer backend"
    ]
    
    best_method = None
    for preferred in preferred_order:
        for method_name, method_args in working_methods:
            if method_name == preferred:
                best_method = (method_name, method_args)
                break
        if best_method:
            break
    
    if not best_method:
        best_method = working_methods[0]  # Use first working method
    
    method_name, method_args = best_method
    print(f"✅ Best method: {method_name}")
    
    # Create configuration
    config = {
        "method": method_name,
        "args": method_args,
        "device_path": device_path
    }
    
    # Save to file
    import json
    with open("external_camera_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print("💾 Saved configuration to: external_camera_config.json")
    
    # Create environment variable suggestions
    env_suggestions = []
    
    if isinstance(method_args, tuple):
        if method_args[1] == cv2.CAP_V4L2:
            if isinstance(method_args[0], str):
                env_suggestions.append(f"STREAM_SOURCE={method_args[0]}")
                env_suggestions.append("OPENCV_BACKEND=V4L2")
            else:
                env_suggestions.append(f"STREAM_SOURCE={method_args[0]}")
                env_suggestions.append("OPENCV_BACKEND=V4L2")
    else:
        env_suggestions.append(f"STREAM_SOURCE={method_args}")
    
    print("\n💡 Add these to your .env file:")
    for suggestion in env_suggestions:
        print(f"   {suggestion}")
    
    return config

def main():
    print("🛠️ External Camera Fix for Bruno Surveillance")
    print("=" * 60)
    
    # Step 1: Check available devices
    video_devices = check_camera_devices()
    
    if not video_devices:
        print("\n❌ No video devices found!")
        print("💡 Make sure your USB camera is plugged in")
        print("💡 Try running: lsusb | grep -i camera")
        return
    
    # Step 2: Check permissions
    fix_camera_permissions()
    
    # Step 3: Test each device
    all_working_methods = {}
    
    for device_path in video_devices:
        # Extract index from device path (e.g., /dev/video0 -> 0)
        try:
            index = int(device_path.split('video')[1])
        except:
            index = None
        
        working_methods = test_camera_access(device_path, index)
        if working_methods:
            all_working_methods[device_path] = working_methods
    
    # Step 4: Create configuration for best device
    if all_working_methods:
        print(f"\n🎉 Found working camera methods!")
        
        # Choose the first working device
        best_device = list(all_working_methods.keys())[0]
        best_methods = all_working_methods[best_device]
        
        print(f"📹 Using device: {best_device}")
        config = create_camera_config(best_methods, best_device)
        
        print(f"\n✅ Camera fix complete!")
        print(f"🚀 Your friend can now use the configuration in external_camera_config.json")
        
    else:
        print(f"\n❌ No working camera methods found!")
        print(f"💡 Troubleshooting steps:")
        print(f"   1. Check if camera is detected: lsusb")
        print(f"   2. Add user to video group: sudo usermod -a -G video $USER")
        print(f"   3. Fix permissions: sudo chmod 666 /dev/video*")
        print(f"   4. Try different USB port")
        print(f"   5. Install v4l-utils: sudo apt-get install v4l-utils")

if __name__ == "__main__":
    main()