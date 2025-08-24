#!/usr/bin/env python3
"""
Test Script for Bruno Bottle Approach System
Tests bottle detection, distance estimation, and movement without full system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import json
import time
from bottle_detection.bottle_detector import BottleDetector
from bottle_detection.distance_estimator import DistanceEstimator

def test_detection_only():
    """Test bottle detection without movement"""
    print("Testing bottle detection system...")
    
    # Load config
    with open('config/bruno_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize components
    detector = BottleDetector(config.get('detection', {}))
    distance_estimator = DistanceEstimator(config.get('distance_estimation', {}))
    
    # Initialize camera
    camera_url = config['camera']['device_id']
    print(f"Connecting to camera: {camera_url}")
    
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    print("✅ Camera connected")
    print("Press 'q' to quit, 's' to save test image")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip frame if configured
        if config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Detect bottles
        bottles, annotated_frame = detector.detect_bottles(frame)
        
        # Add distance information for detected bottles
        if bottles:
            best_bottle = detector.get_best_bottle(bottles)
            frame_height, frame_width = frame.shape[:2]
            
            # Get movement command (for display only)
            movement_command = distance_estimator.get_movement_command(
                best_bottle, frame_width, frame_height
            )
            
            # Draw distance info
            annotated_frame = distance_estimator.draw_distance_info(
                annotated_frame, best_bottle, movement_command
            )
            
            # Print movement command
            if frame_count % 30 == 0:  # Every 30 frames
                distance = movement_command['distance_cm']
                action = movement_command['action']
                print(f"🎯 Best bottle: {distance:.1f}cm - Action: {action}")
        
        # Add test info
        cv2.putText(annotated_frame, "BOTTLE APPROACH TEST - No Movement", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Bottles: {len(bottles)}", 
                   (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Bruno Bottle Approach Test", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"bottle_test_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")
    return True

def main():
    print("Bruno Bottle Approach System Test")
    print("=" * 40)
    
    print("\nThis test will:")
    print("✓ Connect to camera")
    print("✓ Detect plastic bottles")
    print("✓ Estimate distances") 
    print("✓ Show approach commands (but not move)")
    print("\n" + "=" * 40)
    
    success = test_detection_only()
    
    if success:
        print("\n✅ Detection system working!")
        print("\nNext steps:")
        print("1. Run full system: python bruno_main.py")
        print("2. Or web interface: python web_interface.py")
        print("3. Bruno will approach bottles and stop at 1 foot")
    else:
        print("\n❌ Test failed - check camera connection")

if __name__ == "__main__":
    main()