#!/usr/bin/env python3
"""
Servo Test Script - Find the correct head servo ID and positions
"""

import time
from common.ros_robot_controller_sdk import Board

def test_servo(servo_id, positions):
    """Test a specific servo with different positions"""
    board = Board()
    print(f"\nTesting Servo {servo_id}:")
    
    for i, pos in enumerate(positions):
        print(f"  Position {i+1}: {pos}")
        board.pwm_servo_set_position(1.0, [[servo_id, pos]])
        time.sleep(2)

def main():
    print("Servo Testing Script - Find the head servo")
    print("Watch the robot to see which servo moves the head")
    
    # Test common servo positions
    test_positions = [1200, 1500, 1800]  # Left, Center, Right
    
    # Test servos 1-6 (common range for MasterPi)
    for servo_id in range(1, 7):
        try:
            test_servo(servo_id, test_positions)
            input(f"Did servo {servo_id} move the head? Press Enter to continue...")
        except Exception as e:
            print(f"Error testing servo {servo_id}: {e}")

if __name__ == "__main__":
    main()