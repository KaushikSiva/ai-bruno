#!/usr/bin/env python3
"""
Test script for Bruno Obstacle Avoidance System
"""

import cv2
import numpy as np
import time
from bruno_obstacle_avoidance import BrunoObstacleAvoidance

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
    """Test obstacle detection without hardware"""
    print("🧪 Testing Bruno Obstacle Avoidance - Detection Only")
    print("=" * 60)
    
    # Create avoidance system with test config
    config = {
        'obstacle_threshold': 80,
        'danger_threshold': 120,
        'normal_speed': 35,
        'avoidance_speed': 25,
        'camera_url': None  # No camera for testing
    }
    
    avoidance = BrunoObstacleAvoidance(config)
    
    # Test scenarios
    test_scenarios = [
        ("No obstacles", "none"),
        ("Close wall (danger)", "close_wall"),
        ("Obstacle on left", "left_obstacle"),
        ("Obstacle on right", "right_obstacle"),
        ("Far obstacle", "center_far"),
        ("Multiple obstacles", "multiple")
    ]
    
    for scenario_name, obstacle_type in test_scenarios:
        print(f"\n🔍 Testing: {scenario_name}")
        print("-" * 40)
        
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
                print(f"   Obstacle {i+1}: Distance={obs['distance']:.1f}px, Angle={obs['angle']:.1f}°")
        
        # Analyze obstacles and get action plan
        action_plan = avoidance.analyze_obstacles(obstacles)
        print(f"   🤖 Action: {action_plan['action']}")
        print(f"   🔄 Direction: {action_plan['direction']}")
        print(f"   ⚡ Speed: {action_plan['speed']}")
        print(f"   💬 Message: {action_plan['message']}")
        
        # Show visualization
        display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
        cv2.imshow(f"Test: {scenario_name}", display_frame)
        cv2.waitKey(1500)  # Show for 1.5 seconds
        
    cv2.destroyAllWindows()
    print("\n✅ Obstacle detection test complete!")

def test_interactive():
    """Interactive test - press keys to test different scenarios"""
    print("\n🎮 Interactive Bruno Avoidance Test")
    print("=" * 40)
    print("Press keys to test scenarios:")
    print("1 - No obstacles")
    print("2 - Close wall (danger)")
    print("3 - Left obstacle")
    print("4 - Right obstacle")
    print("5 - Far obstacle")
    print("6 - Multiple obstacles")
    print("q - Quit")
    
    config = {
        'obstacle_threshold': 80,
        'danger_threshold': 120,
        'normal_speed': 35,
        'avoidance_speed': 25
    }
    
    avoidance = BrunoObstacleAvoidance(config)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            test_scenario_interactive(avoidance, "none", "No obstacles")
        elif key == ord('2'):
            test_scenario_interactive(avoidance, "close_wall", "Close wall (danger)")
        elif key == ord('3'):
            test_scenario_interactive(avoidance, "left_obstacle", "Left obstacle")
        elif key == ord('4'):
            test_scenario_interactive(avoidance, "right_obstacle", "Right obstacle")
        elif key == ord('5'):
            test_scenario_interactive(avoidance, "center_far", "Far obstacle")
        elif key == ord('6'):
            test_scenario_interactive(avoidance, "multiple", "Multiple obstacles")
    
    cv2.destroyAllWindows()

def test_scenario_interactive(avoidance, obstacle_type, scenario_name):
    """Test a specific scenario interactively"""
    print(f"\n🔍 Testing: {scenario_name}")
    
    # Create frame
    if obstacle_type == "none":
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    else:
        frame = create_test_frame_with_obstacle(obstacle_type)
    
    # Process frame
    obstacles = avoidance.detect_obstacles_vision(frame)
    action_plan = avoidance.analyze_obstacles(obstacles)
    
    # Print results
    print(f"   Detected {len(obstacles)} obstacles")
    if obstacles:
        for i, obs in enumerate(obstacles):
            print(f"   Obstacle {i+1}: Distance={obs['distance']:.1f}px, Angle={obs['angle']:.1f}°")
    
    print(f"   🤖 Action: {action_plan['action']}")
    print(f"   💬 {action_plan['message']}")
    
    # Show visualization
    display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
    cv2.imshow("Interactive Test", display_frame)

def main():
    """Main test function"""
    print("🤖 Bruno Obstacle Avoidance Test Suite")
    print("=" * 50)
    
    try:
        # Run detection tests
        test_obstacle_detection()
        
        # Ask user if they want interactive test
        print("\n❓ Run interactive test? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            test_interactive()
        
        print("\n✅ All tests completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()