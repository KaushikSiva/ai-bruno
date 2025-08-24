#!/usr/bin/env python3
"""
MasterPi Camera Service Checker
Checks if the camera streaming service is running properly
"""

import subprocess
import requests
import cv2
import time

def check_camera_service():
    """Check if camera streaming service is running"""
    print("Checking MasterPi camera service...")
    
    try:
        # Check if process is running
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        camera_processes = []
        for line in result.stdout.split('\n'):
            if 'camera' in line.lower() or '8080' in line or 'mjpg' in line.lower():
                camera_processes.append(line.strip())
        
        if camera_processes:
            print("✓ Found camera-related processes:")
            for process in camera_processes[:5]:  # Show first 5
                print(f"  {process}")
        else:
            print("✗ No camera streaming processes found")
            
    except Exception as e:
        print(f"Could not check processes: {e}")

def check_camera_url():
    """Check if camera URL responds"""
    urls_to_test = [
        'http://127.0.0.1:8080',
        'http://127.0.0.1:8080?action=stream',
        'http://localhost:8080',
        'http://localhost:8080?action=stream'
    ]
    
    print("\nChecking camera URLs...")
    
    for url in urls_to_test:
        print(f"Testing {url}...", end=" ")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ HTTP {response.status_code}")
                content_type = response.headers.get('content-type', '')
                print(f"  Content-Type: {content_type}")
            else:
                print(f"✗ HTTP {response.status_code}")
        except requests.exceptions.ConnectRefused:
            print("✗ Connection refused")
        except requests.exceptions.Timeout:
            print("✗ Timeout")
        except Exception as e:
            print(f"✗ Error: {e}")

def test_opencv_stream():
    """Test OpenCV camera stream connection"""
    print("\nTesting OpenCV camera stream...")
    
    stream_url = 'http://127.0.0.1:8080?action=stream'
    print(f"Connecting to {stream_url}...")
    
    try:
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print("✗ Could not open camera stream")
            return False
        
        print("✓ Camera stream opened")
        
        # Try to read a few frames
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Frame {i+1}: {w}x{h}")
                
                if i == 0:  # Save first frame
                    cv2.imwrite("masterpi_camera_test.jpg", frame)
                    print("  Saved: masterpi_camera_test.jpg")
            else:
                print(f"✗ Frame {i+1}: Failed to read")
                break
            
            time.sleep(0.1)
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def suggest_fixes():
    """Suggest common fixes"""
    print("\n" + "="*50)
    print("TROUBLESHOOTING SUGGESTIONS:")
    print("="*50)
    
    print("\n1. Start camera service (if not running):")
    print("   sudo systemctl start camera-service")
    print("   # or")
    print("   sudo service mjpg-streamer start")
    
    print("\n2. Check camera hardware:")
    print("   ls /dev/video*")
    print("   v4l2-ctl --list-devices")
    
    print("\n3. Test camera directly:")
    print("   raspistill -o test.jpg  # For Pi camera")
    print("   fswebcam test.jpg       # For USB camera")
    
    print("\n4. Restart camera services:")
    print("   sudo systemctl restart camera-service")
    print("   sudo pkill mjpg_streamer")
    
    print("\n5. Check camera permissions:")
    print("   sudo usermod -a -G video $USER")
    
    print("\n6. Manual camera stream start:")
    print("   mjpg_streamer -i 'input_uvc.so -d /dev/video0' -o 'output_http.so -p 8080'")

def main():
    print("MasterPi Camera Service Checker")
    print("="*40)
    
    # Check service
    check_camera_service()
    
    # Check URLs
    check_camera_url()
    
    # Test OpenCV
    success = test_opencv_stream()
    
    if success:
        print("\n✓ Camera system is working!")
        print("You can now run: python bruno_main.py")
    else:
        print("\n✗ Camera system has issues")
        suggest_fixes()

if __name__ == "__main__":
    main()