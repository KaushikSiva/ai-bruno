#!/usr/bin/env python3
"""
Simple test script for Bruno Obstacle Avoidance System
"""

import cv2
import numpy as np
from bruno_avoidance_standalone import StandaloneObstacleAvoidance

def create_test_frame_with_obstacle(obstacle_type="wall"):
    """Create test frame with simulated obstacles"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    if obstacle_type == "close_wall":
        # Close wall - should trigger emergency stop
        cv2.rectangle(frame, (250, 300), (390, 450), (80, 80, 80), -1)
    elif obstacle_type == "left_obstacle":
        # Obstacle on left side
        cv2.rectangle(frame, (150, 200), (250, 400), (100, 100, 100), -1)
    elif obstacle_type == "right_obstacle":
        # Obstacle on right side
        cv2.rectangle(frame, (390, 200), (490, 400), (100, 100, 100), -1)
    elif obstacle_type == "center_far":
        # Far obstacle in center
        cv2.rectangle(frame, (280, 150), (360, 250), (120, 120, 120), -1)
    elif obstacle_type == "multiple":
        # Multiple obstacles
        cv2.rectangle(frame, (100, 250), (180, 400), (90, 90, 90), -1)  # Left
        cv2.rectangle(frame, (460, 250), (540, 400), (90, 90, 90), -1)  # Right
        cv2.rectangle(frame, (300, 350), (340, 450), (70, 70, 70), -1)  # Center close
    
    return frame

def test_obstacle_detection():
    """Test obstacle detection"""
    print("Testing Bruno Obstacle Avoidance System")
    print("=" * 50)
    
    # Create avoidance system
    config = {
        'obstacle_threshold': 80,
        'danger_threshold': 120,
        'normal_speed': 35,
        'avoidance_speed': 25,
        'camera_url': None,  # No camera for testing
        'detection_area': {
            'top': 0.3,
            'bottom': 0.9,
            'left': 0.1,
            'right': 0.9
        }
    }
    
    avoidance = StandaloneObstacleAvoidance(config)
    
    # Test scenarios
    test_scenarios = [
        ("No obstacles", "none"),
        ("Close wall (DANGER)", "close_wall"),
        ("Obstacle on left", "left_obstacle"),
        ("Obstacle on right", "right_obstacle"),
        ("Far obstacle", "center_far"),
        ("Multiple obstacles", "multiple")
    ]
    
    for scenario_name, obstacle_type in test_scenarios:
        print(f"\nTesting: {scenario_name}")
        print("-" * 30)
        
        # Create test frame
        if obstacle_type == "none":
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        else:
            frame = create_test_frame_with_obstacle(obstacle_type)
        
        # Detect obstacles
        obstacles = avoidance.detect_obstacles_vision(frame)
        print(f"   Detected {len(obstacles)} obstacles")
        
        if obstacles:
            for i, obs in enumerate(obstacles):
                print(f"   Obstacle {i+1}: Distance={obs['distance']:.1f}px, Angle={obs['angle']:.1f} deg")
        
        # Analyze obstacles and get action plan
        action_plan = avoidance.analyze_obstacles(obstacles)
        print(f"   Action: {action_plan['action']}")
        print(f"   Direction: {action_plan['direction']}")
        print(f"   Speed: {action_plan['speed']}%")
        print(f"   Emergency: {action_plan['emergency']}")
        print(f"   Message: {action_plan['message']}")
        
        # Show visualization (briefly)
        display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
        cv2.imshow(f"Test: {scenario_name}", display_frame)
        cv2.waitKey(1500)  # Show for 1.5 seconds
        
    cv2.destroyAllWindows()
    print("\nObstacle detection test complete!")

def main():
    """Main test function"""
    print("Bruno Standalone Obstacle Avoidance Test")
    print("=" * 45)
    
    try:
        test_obstacle_detection()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()