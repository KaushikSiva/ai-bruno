#!/usr/bin/env python3
"""
Quick camera check for Bruno
"""
import cv2

print("Quick Camera Check for Bruno (MasterPi)")
print("=" * 40)

# Test MasterPi camera methods first
camera_sources = [
    ('MasterPi Stream', 'http://127.0.0.1:8080?action=stream'),
    ('MasterPi Alt', 'http://localhost:8080?action=stream'), 
    ('USB Device 0', 0),
    ('USB Device 1', 1),
    ('Pi Camera', -1)
]

for name, source in camera_sources:
    print(f"Testing {name}...", end=" ")
    
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Working ({w}x{h})")
                
                # Save a test frame
                filename = f"test_camera_{name.replace(' ', '_').lower()}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  Saved: {filename}")
                
                cap.release()
                
                # Found a working camera - suggest config
                if isinstance(source, str):
                    print(f"\n✓ Use camera stream URL: {source}")
                    print("  Update config: \"device_id\": \"" + source + "\"")
                else:
                    print(f"\n✓ Use device_id: {source} in config")
                break
            else:
                print("✗ Opens but no frame")
                cap.release()
        else:
            print("✗ Cannot open")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("\n✗ No working cameras found!")
    print("Try running: python src/calibration/camera_test.py")