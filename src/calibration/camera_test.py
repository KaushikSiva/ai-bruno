#!/usr/bin/env python3
"""
Camera Detection and Test Script for Bruno
Helps identify available cameras and test functionality
"""

import cv2
import sys
import os

def test_camera_devices():
    """Test multiple camera device IDs to find working cameras"""
    print("Scanning for available cameras...")
    working_cameras = []
    
    # Test camera IDs 0-10
    for device_id in range(11):
        print(f"Testing camera device {device_id}...", end=" ")
        
        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Working! Resolution: {width}x{height}")
                    working_cameras.append({
                        'id': device_id,
                        'width': width,
                        'height': height
                    })
                else:
                    print("✗ Opens but no frame")
                cap.release()
            else:
                print("✗ Cannot open")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return working_cameras

def test_camera_backends():
    """Test different OpenCV camera backends"""
    print("\nTesting OpenCV backends...")
    
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),  
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Any")
    ]
    
    for backend_id, backend_name in backends:
        print(f"Testing {backend_name} backend...", end=" ")
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✓ Working")
                else:
                    print("✗ Opens but no frame")
                cap.release()
            else:
                print("✗ Cannot open")
        except Exception as e:
            print(f"✗ Error: {e}")

def test_raspberry_pi_camera():
    """Test Raspberry Pi camera module"""
    print("\nTesting Raspberry Pi Camera Module...")
    
    # Try picamera2 (newer)
    try:
        from picamera2 import Picamera2
        print("picamera2 available - testing...", end=" ")
        
        picam2 = Picamera2()
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        picam2.start()
        
        # Capture a frame
        frame = picam2.capture_array()
        picam2.stop()
        
        if frame is not None:
            height, width = frame.shape[:2]
            print(f"✓ Working! Resolution: {width}x{height}")
            return True
        else:
            print("✗ No frame captured")
            
    except ImportError:
        print("picamera2 not available")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Try legacy picamera
    try:
        import picamera
        print("picamera (legacy) available - testing...", end=" ")
        
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.start_preview()
            camera.stop_preview()
        
        print("✓ Working!")
        return True
        
    except ImportError:
        print("picamera not available")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return False

def check_camera_permissions():
    """Check camera device permissions"""
    print("\nChecking camera permissions...")
    
    # Check /dev/video* devices
    video_devices = []
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        print(f"Found video devices: {video_devices}")
        
        for device in video_devices:
            try:
                # Check if readable
                with open(device, 'rb') as f:
                    print(f"✓ {device} is readable")
            except PermissionError:
                print(f"✗ {device} permission denied - try: sudo usermod -a -G video $USER")
            except Exception as e:
                print(f"? {device} - {e}")
    else:
        print("No /dev/video* devices found")
    
    # Check if user is in video group
    try:
        import grp
        video_group = grp.getgrnam('video')
        current_user = os.getenv('USER')
        
        if current_user in video_group.gr_mem:
            print(f"✓ User '{current_user}' is in video group")
        else:
            print(f"✗ User '{current_user}' is NOT in video group")
            print("  Fix: sudo usermod -a -G video $USER (then logout/login)")
    except:
        print("Could not check video group membership")

def interactive_camera_test(device_id):
    """Interactive test of a specific camera"""
    print(f"\nInteractive test of camera {device_id}")
    print("Press 'q' to quit, 's' to save a test image")
    
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"Cannot open camera {device_id}")
        return
    
    # Set some common properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f"Camera {device_id} - Frame {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(f"Camera {device_id} Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"camera_{device_id}_test.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved test image: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Camera {device_id} test completed")

def main():
    print("Bruno Camera Diagnostic Tool")
    print("=" * 40)
    
    # Basic system info
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    
    # Check permissions first
    check_camera_permissions()
    
    # Test USB/V4L2 cameras
    working_cameras = test_camera_devices()
    
    # Test camera backends
    test_camera_backends()
    
    # Test Raspberry Pi camera
    pi_camera_works = test_raspberry_pi_camera()
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    
    if working_cameras:
        print(f"✓ Found {len(working_cameras)} working USB/V4L2 cameras:")
        for cam in working_cameras:
            print(f"  - Device {cam['id']}: {cam['width']}x{cam['height']}")
        
        # Suggest configuration
        best_camera = working_cameras[0]
        print(f"\nRecommended configuration:")
        print(f'  "camera": {{"device_id": {best_camera["id"]}}}')
        
    if pi_camera_works:
        print("✓ Raspberry Pi camera module working")
        print('  Alternative config: "device_id": -1 (for Pi camera)')
    
    if not working_cameras and not pi_camera_works:
        print("✗ No working cameras found!")
        print("\nTroubleshooting steps:")
        print("1. Check camera connections")
        print("2. Enable camera in raspi-config (for Pi camera)")
        print("3. Check permissions: sudo usermod -a -G video $USER")
        print("4. Reboot and try again")
    
    # Offer interactive test
    if working_cameras:
        response = input(f"\nTest camera {working_cameras[0]['id']} interactively? (y/n): ")
        if response.lower() == 'y':
            interactive_camera_test(working_cameras[0]['id'])

if __name__ == "__main__":
    main()