#!/usr/bin/env python3
"""
Calibration script for bottle pickup system
Use this to test and adjust coordinates, servo positions, and detection parameters
"""

import cv2
import numpy as np
import time
from common.ros_robot_controller_sdk import Board
from masterpi_sdk.kinematics_sdk.kinematics.arm_move_ik import *

class CalibrationTool:
    def __init__(self):
        self.board = Board()
        self.arm = ArmIK()
        self.camera = cv2.VideoCapture(0)
        
        # Current test parameters
        self.test_x = 15
        self.test_y = 0
        self.test_z = 10
        self.gripper_servo = 5
        self.gripper_pos = 1500
        
    def test_arm_position(self):
        """Test arm movement to current coordinates"""
        print(f"Testing arm position: ({self.test_x}, {self.test_y}, {self.test_z})")
        try:
            self.arm.setPitchRangeMoving((self.test_x, self.test_y, self.test_z), -90, -95, -65, 1500)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def test_gripper(self):
        """Test gripper at current position"""
        print(f"Testing gripper servo {self.gripper_servo} at position {self.gripper_pos}")
        try:
            self.board.pwm_servo_set_position(0.5, [[self.gripper_servo, self.gripper_pos]])
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def show_camera_with_grid(self):
        """Show camera feed with coordinate grid overlay"""
        print("Camera feed with grid - press 'q' to exit")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Draw grid lines
            for i in range(0, w, 50):
                cv2.line(frame, (i, 0), (i, h), (128, 128, 128), 1)
            for i in range(0, h, 50):
                cv2.line(frame, (0, i), (w, i), (128, 128, 128), 1)
            
            # Draw center crosshair
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 2)
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 2)
            
            # Add coordinate text
            cv2.putText(frame, f"Center: ({w//2}, {h//2})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Calibration Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def interactive_calibration(self):
        """Interactive calibration menu"""
        while True:
            print("\n=== MasterPi Bottle Pickup Calibration ===")
            print(f"Current arm position: ({self.test_x}, {self.test_y}, {self.test_z})")
            print(f"Current gripper: servo {self.gripper_servo}, position {self.gripper_pos}")
            print("\nOptions:")
            print("1. Test current arm position")
            print("2. Adjust X coordinate (+/- 1)")
            print("3. Adjust Y coordinate (+/- 1)")
            print("4. Adjust Z coordinate (+/- 1)")
            print("5. Test gripper")
            print("6. Adjust gripper position (+/- 50)")
            print("7. Show camera with grid")
            print("8. Move to home position")
            print("0. Exit")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':
                self.test_arm_position()
            elif choice == '2':
                direction = input("X: + or - ? ").strip()
                if direction == '+':
                    self.test_x += 1
                elif direction == '-':
                    self.test_x -= 1
                self.test_arm_position()
            elif choice == '3':
                direction = input("Y: + or - ? ").strip()
                if direction == '+':
                    self.test_y += 1
                elif direction == '-':
                    self.test_y -= 1
                self.test_arm_position()
            elif choice == '4':
                direction = input("Z: + or - ? ").strip()
                if direction == '+':
                    self.test_z += 1
                elif direction == '-':
                    self.test_z -= 1
                self.test_arm_position()
            elif choice == '5':
                self.test_gripper()
            elif choice == '6':
                direction = input("Gripper: + or - ? ").strip()
                if direction == '+':
                    self.gripper_pos += 50
                elif direction == '-':
                    self.gripper_pos -= 50
                self.gripper_pos = max(500, min(2500, self.gripper_pos))  # Clamp values
                self.test_gripper()
            elif choice == '7':
                self.show_camera_with_grid()
            elif choice == '8':
                print("Moving to home position...")
                self.arm.setPitchRangeMoving((15, 0, 20), -90, -95, -65, 1000)
                time.sleep(2)
            elif choice == '0':
                break
            else:
                print("Invalid choice!")

def main():
    try:
        calibrator = CalibrationTool()
        calibrator.interactive_calibration()
    except KeyboardInterrupt:
        print("\nCalibration interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()