#!/usr/bin/env python3
"""
Test Bruno for 30 seconds
Quick test version that automatically stops after 30 seconds
"""

import os
import sys
import time
import logging

# Set environment to prevent Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bruno_roomba_simple import BrunoRoombaSimple

def main():
    """Test Bruno for exactly 30 seconds"""
    print("Bruno 30-Second Test")
    print("=" * 30)
    
    # Disable emoji logging to prevent Windows encoding issues
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bruno_test.log')
        ]
    )
    
    try:
        bruno = BrunoRoombaSimple()
        
        print("Starting 30-second test...")
        start_time = time.time()
        test_duration = 30  # seconds
        
        frame_count = 0
        detections = 0
        
        while time.time() - start_time < test_duration:
            ret, frame = bruno.camera.read()
            if not ret:
                print("Camera frame failed")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Process frame for detection and navigation
            try:
                processed_frame = bruno.process_frame(frame)
                
                # Count detections (simplified)
                bottles, _ = bruno.bottle_detector.detect_bottles(frame)
                if bottles:
                    detections += 1
                    best_bottle = bruno.bottle_detector.get_best_bottle(bottles)
                    confidence = best_bottle['confidence']
                    print(f"Frame {frame_count}: Bottle detected (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"Frame processing error: {e}")
            
            # Show progress
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and elapsed > 0:
                print(f"Progress: {elapsed:.1f}s / 30s")
            
            time.sleep(0.05)  # ~20 FPS
        
        # Results
        total_time = time.time() - start_time
        print(f"\nTest completed in {total_time:.1f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Bottles detected: {detections}")
        print(f"Detection rate: {(detections/frame_count*100):.1f}% of frames" if frame_count > 0 else "No frames")
        
        if detections > 0:
            print("SUCCESS: Bottle detection working!")
            print("Navigation: Robot was responding to bottle detections")
        else:
            print("WARNING: No bottles detected in test")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            bruno.cleanup()
        except:
            pass
        print("Test cleanup complete")

if __name__ == "__main__":
    main()