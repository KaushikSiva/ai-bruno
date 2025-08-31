#!/usr/bin/env python3
"""
Test Obstacle Avoidance System
Simple test script to demonstrate Bruno's obstacle avoidance capabilities
"""

import cv2
import numpy as np
import time
import random

def create_test_obstacle_frame(obstacle_type="wall", position="center"):
    """Create a test frame with simulated obstacles"""
    # Create a 640x480 frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    if obstacle_type == "wall":
        if position == "center":
            # Wall in center
            cv2.rectangle(frame, (280, 200), (360, 400), (100, 100, 100), -1)
        elif position == "left":
            # Wall on left side
            cv2.rectangle(frame, (100, 200), (200, 400), (100, 100, 100), -1)
        elif position == "right":
            # Wall on right side
            cv2.rectangle(frame, (440, 200), (540, 400), (100, 100, 100), -1)
        elif position == "close":
            # Close wall (danger zone)
            cv2.rectangle(frame, (300, 350), (340, 450), (50, 50, 50), -1)
    
    elif obstacle_type == "object":
        if position == "center":
            # Large object in center
            cv2.rectangle(frame, (250, 150), (390, 350), (150, 150, 150), -1)
        elif position == "small":
            # Small object
            cv2.rectangle(frame, (300, 200), (340, 280), (120, 120, 120), -1)
    
    elif obstacle_type == "multiple":
        # Multiple obstacles
        cv2.rectangle(frame, (200, 200), (250, 300), (100, 100, 100), -1)  # Left
        cv2.rectangle(frame, (390, 200), (440, 300), (100, 100, 100), -1)  # Right
        cv2.rectangle(frame, (300, 350), (340, 400), (80, 80, 80), -1)     # Close
    
    elif obstacle_type == "none":
        # No obstacles
        pass
    
    return frame

def simulate_obstacle_avoidance():
    """Simulate the obstacle avoidance system"""
    print("🧪 Testing Bruno's Obstacle Avoidance System")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        ("No obstacles", "none", "center"),
        ("Wall ahead", "wall", "center"),
        ("Wall on left", "wall", "left"),
        ("Wall on right", "wall", "right"),
        ("Close obstacle", "wall", "close"),
        ("Large object", "object", "center"),
        ("Small object", "object", "small"),
        ("Multiple obstacles", "multiple", "center")
    ]
    
    for scenario_name, obstacle_type, position in scenarios:
        print(f"\n🔍 Testing: {scenario_name}")
        print("-" * 30)
        
        # Create test frame
        frame = create_test_obstacle_frame(obstacle_type, position)
        
        # Simulate obstacle detection
        obstacles = detect_obstacles_simple(frame)
        
        if obstacles:
            print(f"📊 Detected {len(obstacles)} obstacles:")
            for i, obstacle in enumerate(obstacles):
                print(f"   Obstacle {i+1}: {obstacle['distance']:.1f}px away, {obstacle['angle']:.1f}° angle")
            
            # Analyze and plan avoidance
            plan = analyze_obstacles_simple(obstacles)
            print(f"🤖 Action: {plan['action']}")
            print(f"🔄 Direction: {plan['direction']}")
            print(f"⚡ Speed: {plan['speed']}")
            print(f"💬 Message: {plan['message']}")
        else:
            print("✅ No obstacles detected - continuing forward")
        
        # Show frame (optional)
        cv2.imshow(f"Test: {scenario_name}", frame)
        cv2.waitKey(1000)  # Show for 1 second
    
    cv2.destroyAllWindows()
    print("\n✅ Obstacle avoidance test complete!")

def detect_obstacles_simple(frame):
    """Simple obstacle detection using edge detection"""
    obstacles = []
    H, W = frame.shape[:2]
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small noise
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate distance estimate (simplified)
            distance = estimate_distance_simple(w, h, W, H)
            
            # Calculate angle from center
            center_x = x + w/2
            angle = np.arctan2(center_x - W/2, W/2) * 180 / np.pi
            
            # Only consider obstacles in front of robot (within 60 degrees)
            if abs(angle) < 60:
                obstacle = {
                    'distance': distance,
                    'angle': angle,
                    'confidence': min(area / 10000, 1.0),
                    'type': 'object',
                    'bbox': (x, y, w, h)
                }
                obstacles.append(obstacle)
                
    except Exception as e:
        print(f"Obstacle detection error: {e}")
    
    return obstacles

def estimate_distance_simple(width, height, frame_width, frame_height):
    """Simple distance estimation based on object size"""
    apparent_size = width * height
    max_size = frame_width * frame_height * 0.1
    
    distance_ratio = 1.0 - (apparent_size / max_size)
    distance_ratio = max(0.0, min(1.0, distance_ratio))
    
    return distance_ratio * 300  # Max distance of 300 pixels

def analyze_obstacles_simple(obstacles):
    """Analyze obstacles and determine avoidance strategy"""
    if not obstacles:
        return {
            'action': 'continue',
            'direction': 'forward',
            'speed': 40,
            'emergency_stop': False,
            'message': 'No obstacles detected'
        }
    
    # Find closest obstacle
    closest_obstacle = min(obstacles, key=lambda x: x['distance'])
    distance = closest_obstacle['distance']
    angle = closest_obstacle['angle']
    
    # Determine action based on distance
    if distance < 80:  # Danger zone
        return {
            'action': 'emergency_stop',
            'direction': 'stop',
            'speed': 0,
            'emergency_stop': True,
            'message': f'DANGER: Obstacle at {distance:.1f}px'
        }
    elif distance < 120:  # Caution zone
        return plan_avoidance_simple(closest_obstacle)
    else:  # Safe zone
        return {
            'action': 'continue',
            'direction': 'forward',
            'speed': 40,
            'emergency_stop': False,
            'message': f'Safe: Obstacle at {distance:.1f}px'
        }

def plan_avoidance_simple(obstacle):
    """Plan avoidance strategy based on obstacle position"""
    angle = obstacle['angle']
    
    if abs(angle) < 20:  # Obstacle directly ahead
        # Choose left or right based on which side has more space
        if angle > 0:  # Slightly to the right
            return {
                'action': 'avoid',
                'direction': 'arc_left',
                'speed': 25,
                'emergency_stop': False,
                'message': f'AVOID: Obstacle ahead, turning left'
            }
        else:  # Slightly to the left
            return {
                'action': 'avoid',
                'direction': 'arc_right',
                'speed': 25,
                'emergency_stop': False,
                'message': f'AVOID: Obstacle ahead, turning right'
            }
    elif angle > 20:  # Obstacle to the right
        return {
            'action': 'avoid',
            'direction': 'arc_left',
            'speed': 30,
            'emergency_stop': False,
            'message': f'AVOID: Obstacle to right, turning left'
        }
    else:  # Obstacle to the left
        return {
            'action': 'avoid',
            'direction': 'arc_right',
            'speed': 30,
            'emergency_stop': False,
            'message': f'AVOID: Obstacle to left, turning right'
        }

def interactive_test():
    """Interactive test where user can create obstacles"""
    print("\n🎮 Interactive Obstacle Avoidance Test")
    print("Press keys to test different scenarios:")
    print("1 - No obstacles")
    print("2 - Wall ahead")
    print("3 - Wall on left")
    print("4 - Wall on right")
    print("5 - Close obstacle")
    print("6 - Large object")
    print("7 - Multiple obstacles")
    print("q - Quit")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            frame = create_test_obstacle_frame("none", "center")
            test_scenario(frame, "No obstacles")
        elif key == ord('2'):
            frame = create_test_obstacle_frame("wall", "center")
            test_scenario(frame, "Wall ahead")
        elif key == ord('3'):
            frame = create_test_obstacle_frame("wall", "left")
            test_scenario(frame, "Wall on left")
        elif key == ord('4'):
            frame = create_test_obstacle_frame("wall", "right")
            test_scenario(frame, "Wall on right")
        elif key == ord('5'):
            frame = create_test_obstacle_frame("wall", "close")
            test_scenario(frame, "Close obstacle")
        elif key == ord('6'):
            frame = create_test_obstacle_frame("object", "center")
            test_scenario(frame, "Large object")
        elif key == ord('7'):
            frame = create_test_obstacle_frame("multiple", "center")
            test_scenario(frame, "Multiple obstacles")
    
    cv2.destroyAllWindows()

def test_scenario(frame, scenario_name):
    """Test a specific scenario and show results"""
    print(f"\n🔍 Testing: {scenario_name}")
    
    # Detect obstacles
    obstacles = detect_obstacles_simple(frame)
    
    if obstacles:
        print(f"📊 Detected {len(obstacles)} obstacles:")
        for i, obstacle in enumerate(obstacles):
            print(f"   Obstacle {i+1}: {obstacle['distance']:.1f}px away, {obstacle['angle']:.1f}° angle")
        
        # Analyze and plan avoidance
        plan = analyze_obstacles_simple(obstacles)
        print(f"🤖 Action: {plan['action']}")
        print(f"🔄 Direction: {plan['direction']}")
        print(f"💬 Message: {plan['message']}")
    else:
        print("✅ No obstacles detected - continuing forward")
    
    # Show frame
    cv2.imshow("Interactive Test", frame)

if __name__ == "__main__":
    print("🤖 Bruno Obstacle Avoidance Test Suite")
    print("=" * 50)
    
    # Run automated tests
    simulate_obstacle_avoidance()
    
    # Run interactive test
    interactive_test()
    
    print("\n✅ All tests complete!")
