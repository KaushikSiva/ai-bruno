#!/usr/bin/env python3
"""
Camera Diagnostics for Windows
Comprehensive camera detection and troubleshooting
"""

import cv2
import sys
import os
import subprocess

def check_system_cameras():
    """Check system cameras using Windows commands"""
    print("=== System Camera Check ===")
    
    try:
        # Use PowerShell to list camera devices
        cmd = ['powershell', '-Command', 
               'Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status, InstanceId']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("Windows Camera Devices:")
            print(result.stdout)
        else:
            print("No cameras found by Windows device manager")
            
    except Exception as e:
        print(f"Could not run PowerShell command: {e}")
    
    try:
        # Alternative method using WMI
        cmd = ['wmic', 'path', 'Win32_PnPEntity', 'where', 
               '"Name like \'%camera%\' or Name like \'%webcam%\'"', 
               'get', 'Name,Status']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            print("\nWMI Camera Devices:")
            print(result.stdout)
            
    except Exception as e:
        print(f"Could not run WMI command: {e}")

def test_opencv_backends():
    """Test different OpenCV backends"""
    print("\n=== OpenCV Backend Test ===")
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name} backend...")
        
        for device_id in range(3):
            try:
                print(f"  Device {device_id}...", end=" ")
                cap = cv2.VideoCapture(device_id, backend_id)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"[OK] {width}x{height}")
                        cap.release()
                        return True  # Found working camera
                    else:
                        print("[FAIL] No frame")
                        cap.release()
                else:
                    print("[FAIL] Cannot open")
                    
            except Exception as e:
                print(f"[ERROR] {e}")
    
    return False

def check_camera_permissions():
    """Check if camera access is allowed"""
    print("\n=== Camera Permissions Check ===")
    
    # Check Windows privacy settings (requires registry access)
    try:
        import winreg
        
        # Check global camera access
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
                if value == "Allow":
                    print("[OK] Global camera access allowed")
                else:
                    print(f"[WARNING] Global camera access: {value}")
        except:
            print("[INFO] Could not check global camera permissions")
        
    except ImportError:
        print("[INFO] winreg not available for permission check")

def test_camera_apps():
    """Check if common apps might be using the camera"""
    print("\n=== Camera Usage Check ===")
    
    # Common applications that might use camera
    camera_apps = [
        "Teams.exe", "Zoom.exe", "Skype.exe", "chrome.exe", 
        "msedge.exe", "firefox.exe", "obs64.exe", "obs32.exe"
    ]
    
    try:
        # Get running processes
        cmd = ['tasklist', '/FO', 'CSV']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            running_processes = result.stdout.lower()
            found_apps = []
            
            for app in camera_apps:
                if app.lower() in running_processes:
                    found_apps.append(app)
            
            if found_apps:
                print(f"[WARNING] Apps that might be using camera: {', '.join(found_apps)}")
                print("Try closing these applications and test again")
            else:
                print("[OK] No common camera applications detected")
                
    except Exception as e:
        print(f"Could not check running processes: {e}")

def provide_solutions():
    """Provide troubleshooting solutions"""
    print("\n=== Troubleshooting Steps ===")
    print("1. Physical Connection:")
    print("   - Ensure USB camera is properly plugged in")
    print("   - Try a different USB port")
    print("   - Check if camera LED/indicator is on")
    
    print("\n2. Windows Settings:")
    print("   - Go to Settings > Privacy & Security > Camera")
    print("   - Enable 'Camera access for this device'")
    print("   - Enable 'Let apps access your camera'")
    
    print("\n3. Device Manager:")
    print("   - Press Win+X and select 'Device Manager'")
    print("   - Look under 'Cameras' or 'Imaging devices'")
    print("   - If you see warning signs, right-click and 'Update driver'")
    
    print("\n4. Application Conflicts:")
    print("   - Close any video calling apps (Teams, Zoom, Skype)")
    print("   - Close any browsers that might be using camera")
    print("   - Close any streaming software (OBS, etc.)")
    
    print("\n5. Camera Testing:")
    print("   - Test camera in Windows Camera app first")
    print("   - If Windows Camera app works, the issue is with OpenCV/Python")

def main():
    print("Camera Diagnostics for Bruno External Camera")
    print("=" * 50)
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    
    # Run diagnostics
    check_system_cameras()
    found_camera = test_opencv_backends()
    check_camera_permissions()
    test_camera_apps()
    provide_solutions()
    
    print("\n" + "=" * 50)
    if found_camera:
        print("[OK] At least one camera was detected!")
        print("Try running the camera test again.")
    else:
        print("[INFO] No cameras detected through OpenCV")
        print("Follow the troubleshooting steps above.")
    
    print("\nNext steps:")
    print("1. Connect your external camera if not already connected")
    print("2. Follow the troubleshooting steps above")
    print("3. Run: python test_camera_streaming.py")
    print("4. Run: python setup_external_camera.py")

if __name__ == "__main__":
    main()