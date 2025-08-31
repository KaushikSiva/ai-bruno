#!/usr/bin/env python3
"""
Test script for Standalone Bruno Obstacle Avoidance System
"""

import cv2
import numpy as np
import time
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
    elif obstacle_type == "small_objects":
        # Small objects scattered
        cv2.rectangle(frame, (200, 300), (240, 350), (100, 100, 100), -1)
        cv2.rectangle(frame, (400, 280), (450, 330), (100, 100, 100), -1)
        cv2.rectangle(frame, (320, 320), (360, 380), (100, 100, 100), -1)
    
    return frame

def test_obstacle_detection():
    """Test obstacle detection without hardware"""
    print("🧪 Testing Standalone Bruno Obstacle Avoidance")
    print("=" * 60)
    
    # Create avoidance system with test config
    config = {
        'obstacle_threshold': 80,
        'danger_threshold': 120,
        'normal_speed': 35,
        'avoidance_speed': 25,
        'camera_url': None  # No camera for testing
    }
    
    avoidance = StandaloneObstacleAvoidance(config)
    
    # Test scenarios
    test_scenarios = [
        ("No obstacles", "none"),
        ("Close wall (DANGER)", "close_wall"),
        ("Obstacle on left", "left_obstacle"),
        ("Obstacle on right", "right_obstacle"),
        ("Far obstacle", "center_far"),
        ("Multiple obstacles", "multiple"),
        ("Small scattered objects", "small_objects")
    ]
    
    for scenario_name, obstacle_type in test_scenarios:
        print(f"\nTesting: {scenario_name}")
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
                print(f"   Obstacle {i+1}: Distance={obs['distance']:.1f}px, Angle={obs['angle']:.1f}°, Area={obs['area']:.0f}px²")
        
        # Analyze obstacles and get action plan
        action_plan = avoidance.analyze_obstacles(obstacles)
        print(f"   🤖 Action: {action_plan['action']}")
        print(f"   🔄 Direction: {action_plan['direction']}")
        print(f"   Speed: {action_plan['speed']}%")
        print(f"   🚨 Emergency: {action_plan['emergency']}")
        print(f"   💬 Message: {action_plan['message']}")
        
        # Simulate action execution
        avoidance.execute_avoidance_action(action_plan)
        
        # Show visualization
        display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
        cv2.imshow(f"Test: {scenario_name}", display_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        
    cv2.destroyAllWindows()
    print("\nPASS: Obstacle detection test complete!")

def test_interactive():
    """Interactive test - press keys to test different scenarios"""
    print("\n🎮 Interactive Bruno Avoidance Test")
    print("=" * 40)
    print("Press keys to test scenarios:")
    print("1 - No obstacles")
    print("2 - Close wall (DANGER)")
    print("3 - Left obstacle")
    print("4 - Right obstacle")
    print("5 - Far obstacle")
    print("6 - Multiple obstacles")
    print("7 - Small scattered objects")
    print("r - Run continuous random test")
    print("q - Quit")
    
    config = {
        'obstacle_threshold': 80,
        'danger_threshold': 120,
        'normal_speed': 35,
        'avoidance_speed': 25
    }
    
    avoidance = StandaloneObstacleAvoidance(config)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            test_scenario_interactive(avoidance, "none", "No obstacles")
        elif key == ord('2'):
            test_scenario_interactive(avoidance, "close_wall", "Close wall (DANGER)")
        elif key == ord('3'):
            test_scenario_interactive(avoidance, "left_obstacle", "Left obstacle")
        elif key == ord('4'):
            test_scenario_interactive(avoidance, "right_obstacle", "Right obstacle")
        elif key == ord('5'):
            test_scenario_interactive(avoidance, "center_far", "Far obstacle")
        elif key == ord('6'):
            test_scenario_interactive(avoidance, "multiple", "Multiple obstacles")
        elif key == ord('7'):
            test_scenario_interactive(avoidance, "small_objects", "Small scattered objects")
        elif key == ord('r'):
            run_continuous_test(avoidance)
            break
    
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
    print(f"   📊 Detected {len(obstacles)} obstacles")
    if obstacles:
        for i, obs in enumerate(obstacles):
            print(f"   Obstacle {i+1}: Distance={obs['distance']:.1f}px, Angle={obs['angle']:.1f}°")
    
    print(f"   🤖 Action: {action_plan['action']}")
    print(f"   🔄 Direction: {action_plan['direction']}")
    print(f"   💬 {action_plan['message']}")
    
    # Simulate execution
    avoidance.execute_avoidance_action(action_plan)
    
    # Show visualization
    display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
    cv2.imshow("Interactive Test", display_frame)

def run_continuous_test(avoidance):
    """Run continuous test with changing scenarios"""
    import random
    
    print("🔄 Running continuous random test - Press ESC to stop")
    
    scenarios = ["none", "close_wall", "left_obstacle", "right_obstacle", "center_far", "multiple", "small_objects"]
    scenario_names = [
        "No obstacles", "Close wall (DANGER)", "Left obstacle", 
        "Right obstacle", "Far obstacle", "Multiple obstacles", "Small scattered objects"
    ]
    
    frame_count = 0
    
    while True:
        # Change scenario every 3 seconds (30 frames at 10 FPS)
        if frame_count % 30 == 0:
            current_scenario = random.choice(scenarios)
            current_name = scenario_names[scenarios.index(current_scenario)]
            print(f"🎲 Switching to: {current_name}")
        
        # Create frame
        if current_scenario == "none":
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        else:
            frame = create_test_frame_with_obstacle(current_scenario)
        
        # Process frame
        obstacles = avoidance.detect_obstacles_vision(frame)
        action_plan = avoidance.analyze_obstacles(obstacles)
        avoidance.execute_avoidance_action(action_plan)
        
        # Draw and display
        display_frame = avoidance.draw_obstacle_info(frame, obstacles, action_plan)
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {frame_count}, Scenario: {current_name}", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Continuous Test", display_frame)
        
        # Check for exit
        key = cv2.waitKey(100) & 0xFF  # 10 FPS
        if key == 27:  # ESC
            break
        
        frame_count += 1
    
    cv2.destroyAllWindows()

def main():
    """Main test function"""
    print("Bruno Standalone Obstacle Avoidance Test Suite")
    print("=" * 60)
    print("This test demonstrates Bruno's obstacle detection and avoidance")
    print("capabilities using computer vision without hardware dependencies.")
    print("=" * 60)
    
    try:
        # Run detection tests
        test_obstacle_detection()
        
        # Ask user if they want interactive test
        print("\n❓ Run interactive test? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            test_interactive()
        
        print("\nPASS: All tests completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"ERROR: Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()