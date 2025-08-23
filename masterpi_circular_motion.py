#!/usr/bin/env python3
"""
Hiwonder Master PI Robot - Circular Motion with Head Nodding
Makes the robot move in a circular pattern while nodding its head sideways
"""

import time
import math
import threading
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board

class MasterPiCircularMotion:
    def __init__(self):
        self.chassis = mecanum.MecanumChassis()
        self.board = Board()
        self.running = False
        
        # Head servo configuration - try different IDs: 2, 3, 4, 5, or 6
        self.head_servo_id = 2  # Change this to the correct servo ID
        self.head_center_position = 1500  # Center position pulse width
        self.head_left_position = 1200    # Left position pulse width
        self.head_right_position = 1800   # Right position pulse width
        
        # Motion parameters
        self.circular_speed = 30          # Forward speed for circular motion
        self.rotation_speed = 0.5         # Rotation speed for circular motion
        self.head_nod_duration = 1.0      # Time for each head movement
        
    def start_circular_motion(self):
        """Start the circular motion of the robot"""
        print("Starting circular motion...")
        self.chassis.set_velocity(self.circular_speed, 90, self.rotation_speed)
        
    def stop_motion(self):
        """Stop all robot motion"""
        print("Stopping motion...")
        self.chassis.set_velocity(0, 0, 0)
        # Return head to center position
        self.board.pwm_servo_set_position(0.5, [[self.head_servo_id, self.head_center_position]])
        
    def nod_head_sideways(self):
        """Make the robot nod its head sideways continuously"""
        while self.running:
            # Nod left
            self.board.pwm_servo_set_position(self.head_nod_duration, 
                                            [[self.head_servo_id, self.head_left_position]])
            time.sleep(self.head_nod_duration)
            
            if not self.running:
                break
                
            # Nod right
            self.board.pwm_servo_set_position(self.head_nod_duration, 
                                            [[self.head_servo_id, self.head_right_position]])
            time.sleep(self.head_nod_duration)
            
    def execute_motion(self, duration=30):
        """
        Execute the combined circular motion and head nodding
        
        Args:
            duration (float): How long to run the motion in seconds
        """
        print(f"Starting circular motion with head nodding for {duration} seconds...")
        
        self.running = True
        
        # Start circular motion
        self.start_circular_motion()
        
        # Start head nodding in a separate thread
        head_thread = threading.Thread(target=self.nod_head_sideways)
        head_thread.daemon = True
        head_thread.start()
        
        # Run for specified duration
        time.sleep(duration)
        
        # Stop motion
        self.running = False
        self.stop_motion()
        
        # Wait for head thread to finish
        head_thread.join(timeout=2)
        
        print("Motion completed!")

def main():
    """Main function to run the circular motion demo"""
    try:
        robot = MasterPiCircularMotion()
        
        print("Hiwonder Master PI - Circular Motion with Head Nodding Demo")
        print("Press Ctrl+C to stop the program")
        
        # Initialize head to center position
        robot.board.pwm_servo_set_position(0.5, [[robot.head_servo_id, robot.head_center_position]])
        time.sleep(1)
        
        # Execute the motion for 30 seconds
        robot.execute_motion(duration=30)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        if 'robot' in locals():
            robot.running = False
            robot.stop_motion()
    except Exception as e:
        print(f"Error: {e}")
        if 'robot' in locals():
            robot.running = False
            robot.stop_motion()

if __name__ == "__main__":
    main()