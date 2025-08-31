#!/usr/bin/env python3
"""
Windows Camera Detection for Bruno
Detects available cameras on Windows system
"""

import cv2
import sys

def test_cameras():
    """Test cameras with Windows-specific methods"""
    print("Windows Camera Detection for Bruno")
    print("=" * 40)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    print()
    
    working_cameras = []
    
    # Test DirectShow backend (Windows default)
    print("Testing cameras with DirectShow backend...")
    for device_id in range(10):
        print(f"Testing camera {device_id}...", end=" ")
        
        try:
            # Try DirectShow backend specifically
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"[OK] {width}x{height}")
                    
                    # Get additional info
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    
                    working_cameras.append({
                        'device_id': device_id,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'fourcc': fourcc
                    })
                else:
                    print("[FAIL] No frame")
                cap.release()
            else:
                print("[FAIL] Cannot open")
                
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print()
    if working_cameras:
        print(f"Found {len(working_cameras)} working cameras:")
        for cam in working_cameras:
            print(f"  Device {cam['device_id']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
        
        # Test the first camera interactively
        print("\nTesting first camera...")
        test_camera = working_cameras[0]
        
        cap = cv2.VideoCapture(test_camera['device_id'], cv2.CAP_DSHOW)
        if cap.isOpened():
            print("Press ESC to exit camera test")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                cv2.putText(frame, f"Camera {test_camera['device_id']} - Frame {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press ESC to exit", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(f"Camera {test_camera['device_id']} Test", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Save a test image
            cap = cv2.VideoCapture(test_camera['device_id'], cv2.CAP_DSHOW)
            ret, frame = cap.read()
            if ret:
                filename = f"camera_{test_camera['device_id']}_test.jpg"
                cv2.imwrite(filename, frame)
                print(f"Test image saved: {filename}")
            cap.release()
        
        return working_cameras
    else:
        print("No working cameras found!")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is plugged in")
        print("2. Check if the camera is being used by another application")
        print("3. Try a different USB port")
        print("4. Check Windows Device Manager for camera devices")
        return []

if __name__ == "__main__":
    test_cameras()