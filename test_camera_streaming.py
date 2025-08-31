#!/usr/bin/env python3
"""
Camera Streaming Test for Windows
Tests if your external camera can stream video properly
"""

import cv2
import sys
import time

def test_camera_streaming():
    """Test camera streaming with live preview"""
    print("Camera Streaming Test")
    print("=" * 30)
    
    # Detect cameras first
    working_cameras = []
    print("Detecting cameras...")
    
    for device_id in range(5):
        try:
            print(f"Testing camera {device_id}...", end=" ")
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"[OK] {width}x{height}")
                    working_cameras.append(device_id)
                    cap.release()
                else:
                    print("[FAIL] No frame")
                    cap.release()
            else:
                print("[FAIL] Cannot open")
                
        except Exception as e:
            print(f"[ERROR] {e}")
    
    if not working_cameras:
        print("\n[ERROR] No working cameras found!")
        print("Make sure your external camera is connected and not in use by another app.")
        return False
    
    print(f"\nFound cameras: {working_cameras}")
    
    # Test streaming from each camera
    for camera_id in working_cameras:
        print(f"\n--- Testing Camera {camera_id} ---")
        
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            continue
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera {camera_id} properties: {width}x{height} @ {fps:.1f}fps")
        
        # Test streaming
        print(f"Starting stream test for camera {camera_id}...")
        print("Press 'q' to stop, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"[ERROR] Failed to read frame from camera {camera_id}")
                break
            
            frame_count += 1
            
            # Add info overlay
            cv2.putText(frame, f"Camera {camera_id} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(f"Camera {camera_id} Stream Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"[OK] Camera {camera_id} streaming test completed")
                break
            elif key == ord('s'):
                filename = f"camera_{camera_id}_screenshot.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show statistics
        elapsed = time.time() - start_time
        if elapsed > 0:
            avg_fps = frame_count / elapsed
            print(f"Camera {camera_id} stats: {frame_count} frames in {elapsed:.1f}s (avg {avg_fps:.1f} fps)")
        
        # Ask if user wants to test next camera
        if camera_id != working_cameras[-1]:
            choice = input(f"\nTest next camera? (y/n): ").lower()
            if choice != 'y':
                break
    
    return True

def main():
    try:
        success = test_camera_streaming()
        if success:
            print("\n[OK] Camera streaming test completed!")
        else:
            print("\n[ERROR] Camera streaming test failed!")
    except KeyboardInterrupt:
        print("\n[INFO] Test cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()