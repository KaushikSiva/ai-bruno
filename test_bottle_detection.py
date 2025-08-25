#!/usr/bin/env python3
"""
Test bottle detection without movement
Quick test to see if bottles are being detected properly
"""

import os
import sys
import cv2
import json

# Set environment
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bottle_detection.bottle_detector import BottleDetector
from bottle_detection.distance_estimator import DistanceEstimator

def test_detection():
    """Test bottle detection with camera"""
    print("🧪 Testing Bottle Detection")
    print("=" * 30)
    
    # Load or create config
    config_file = "config/bruno_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("✅ Loaded config file")
    else:
        print("⚠️  No config file, using defaults")
        config = {
            'detection': {
                'color_detection': {
                    'clear_plastic': {
                        'lower_hsv': [0, 0, 150],
                        'upper_hsv': [180, 50, 255]
                    }
                },
                'size_filter': {
                    'min_area': 1000,
                    'max_area': 50000,
                    'min_aspect_ratio': 1.2,
                    'max_aspect_ratio': 4.5
                },
                'morphology': {
                    'kernel_size': 3,
                    'iterations': 2
                },
                'confidence_threshold': 0.6
            },
            'distance_estimation': {
                'focal_length': 500,
                'real_bottle_height': 20,
                'stop_distance': 30,
                'approach_distance': 100
            }
        }
    
    # Initialize detector
    try:
        detector = BottleDetector(config['detection'])
        estimator = DistanceEstimator(config['distance_estimation'])
        print("✅ Detectors initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detectors: {e}")
        return
    
    # Try to connect to camera
    camera_sources = [
        'http://127.0.0.1:8080?action=stream',
        'http://localhost:8080?action=stream', 
        0, 1
    ]
    
    camera = None
    for source in camera_sources:
        print(f"🔍 Trying camera: {source}")
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera = cap
                    print(f"✅ Camera connected: {source}")
                    break
                else:
                    cap.release()
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    if not camera:
        print("❌ No camera found")
        return
    
    print("\n🎥 Starting detection test...")
    print("📱 Processing frames (headless mode)")
    print("⏹️  This will run for 30 seconds")
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        import time
        while time.time() - start_time < 30:  # Run for 30 seconds
            ret, frame = camera.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Detect bottles
            bottles, annotated_frame = detector.detect_bottles(frame)
            
            if bottles:
                detection_count += 1
                best_bottle = detector.get_best_bottle(bottles)
                
                # Get movement command
                frame_height, frame_width = frame.shape[:2]
                movement_cmd = estimator.get_movement_command(best_bottle, frame_width, frame_height)
                
                print(f"🎯 Frame {frame_count}: {len(bottles)} bottles detected!")
                print(f"   Best bottle confidence: {best_bottle['confidence']:.2f}")
                print(f"   Distance: {movement_cmd['distance_cm']:.1f}cm")
                print(f"   Action: {movement_cmd['action']}")
                
                # Save detection image
                filename = f"test_detection_{frame_count}.jpg"
                annotated_frame = estimator.draw_distance_info(annotated_frame, best_bottle, movement_cmd)
                cv2.imwrite(filename, annotated_frame)
                print(f"   💾 Saved: {filename}")
            
            elif frame_count % 60 == 0:  # Log every 60 frames (~3 seconds)
                print(f"📊 Frame {frame_count}: No bottles detected")
            
            time.sleep(0.05)  # ~20 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    
    finally:
        camera.release()
        
        print(f"\n📊 Test Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Detections: {detection_count}")
        
        if detection_count > 0:
            print("✅ Bottle detection is working!")
            print("💡 Try running: python bruno_simple.py")
        else:
            print("❌ No bottles detected")
            print("💡 Try:")
            print("   - Place a plastic bottle in camera view")
            print("   - Adjust lighting")
            print("   - Check camera focus")

if __name__ == "__main__":
    test_detection()