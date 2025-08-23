#!/usr/bin/env python3
"""
Hiwonder Master PI Robot - Plastic Bottle Detection and Pickup
Uses OpenCV for bottle detection and robotic arm for pickup
"""

import cv2
import numpy as np
import time
import math
import threading
from common.ros_robot_controller_sdk import Board
from masterpi_sdk.kinematics_sdk.kinematics.ArmIK import ArmIK

class BottlePickupRobot:
    def __init__(self):
        # Initialize hardware
        self.board = Board()
        self.arm = ArmIK()
        self.camera = cv2.VideoCapture(0)
        
        # Camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Arm configuration
        self.gripper_servo_id = 5  # Adjust based on your robot
        self.gripper_open_pos = 1800
        self.gripper_closed_pos = 1200
        
        # Detection parameters
        self.detection_running = False
        self.bottle_detected = False
        self.bottle_center = None
        
        # Arm positions
        self.home_position = (15, 0, 20)  # Safe home position
        self.pickup_height = 5  # Height to pickup bottles
        self.drop_position = (20, -15, 10)  # Where to drop bottles
        
        # Initialize arm to home position
        self.move_to_home()
        self.open_gripper()
        
    def move_to_home(self):
        """Move arm to home position"""
        print("Moving to home position...")
        self.arm.setPitchRangeMoving(self.home_position, -90, -95, -65, 1000)
        time.sleep(1.5)
        
    def open_gripper(self):
        """Open the gripper"""
        self.board.pwm_servo_set_position(0.5, [[self.gripper_servo_id, self.gripper_open_pos]])
        time.sleep(0.5)
        
    def close_gripper(self):
        """Close the gripper"""
        self.board.pwm_servo_set_position(0.5, [[self.gripper_servo_id, self.gripper_closed_pos]])
        time.sleep(0.5)
        
    def detect_bottles(self, frame):
        """
        Detect plastic bottles in the frame using shape and color detection
        Returns the center coordinates of detected bottles
        """
        # Convert to HSV for better color detection
        hsv = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        
        # Define range for bottle colors (clear/white plastic)
        # Adjust these ranges based on your bottles
        lower_bottle = np.array([0, 0, 150])
        upper_bottle = np.array([180, 50, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bottle, upper_bottle)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bottles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust based on bottle size and distance)
            if area > 1000:
                # Check if shape is bottle-like (elongated)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                
                # Bottles are typically taller than wide
                if 1.5 <= aspect_ratio <= 4.0:
                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    bottles.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
                    
                    # Draw detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, "Bottle", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return bottles, frame
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y, frame_width=640, frame_height=480):
        """
        Convert pixel coordinates to world coordinates for the arm
        This is a simplified conversion - you may need to calibrate for your setup
        """
        # Camera field of view approximation
        # Adjust these values based on your camera calibration
        fov_x = 60  # degrees
        fov_y = 45  # degrees
        
        # Convert to normalized coordinates (-1 to 1)
        norm_x = (pixel_x - frame_width / 2) / (frame_width / 2)
        norm_y = (pixel_y - frame_height / 2) / (frame_height / 2)
        
        # Convert to world coordinates (approximate)
        # These values need calibration for your specific setup
        world_x = 15 + norm_x * 10  # Forward distance
        world_y = norm_y * 8        # Left/right
        world_z = self.pickup_height
        
        return (world_x, world_y, world_z)
    
    def pickup_bottle(self, bottle_position):
        """
        Pick up a bottle at the given world coordinates
        """
        print(f"Attempting to pick up bottle at {bottle_position}")
        
        try:
            # Move to position above the bottle
            approach_pos = (bottle_position[0], bottle_position[1], bottle_position[2] + 5)
            print("Moving to approach position...")
            self.arm.setPitchRangeMoving(approach_pos, -90, -95, -65, 1500)
            time.sleep(2)
            
            # Lower to pickup position
            print("Lowering to pickup position...")
            self.arm.setPitchRangeMoving(bottle_position, -90, -95, -65, 1000)
            time.sleep(1.5)
            
            # Close gripper to grab bottle
            print("Closing gripper...")
            self.close_gripper()
            time.sleep(1)
            
            # Lift the bottle
            print("Lifting bottle...")
            lift_pos = (bottle_position[0], bottle_position[1], bottle_position[2] + 8)
            self.arm.setPitchRangeMoving(lift_pos, -90, -95, -65, 1000)
            time.sleep(1.5)
            
            # Move to drop position
            print("Moving to drop position...")
            self.arm.setPitchRangeMoving(self.drop_position, -90, -95, -65, 1500)
            time.sleep(2)
            
            # Release bottle
            print("Releasing bottle...")
            self.open_gripper()
            time.sleep(1)
            
            # Return to home
            print("Returning to home position...")
            self.move_to_home()
            
            return True
            
        except Exception as e:
            print(f"Error during pickup: {e}")
            self.move_to_home()
            return False
    
    def run_detection_loop(self):
        """
        Main detection and pickup loop
        """
        print("Starting bottle detection and pickup system...")
        print("Press 'q' to quit, 'space' to manually trigger pickup")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect bottles
            bottles, annotated_frame = self.detect_bottles(frame)
            
            # Display frame
            cv2.putText(annotated_frame, f"Bottles detected: {len(bottles)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press SPACE to pickup, Q to quit", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Bottle Detection", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Check for user input
            if key == ord('q'):
                break
            elif key == ord(' ') and bottles:
                # Pick up the largest bottle
                largest_bottle = max(bottles, key=lambda b: b['area'])
                pixel_coords = largest_bottle['center']
                world_coords = self.pixel_to_world_coordinates(pixel_coords[0], pixel_coords[1])
                
                print(f"Picking up bottle at pixel coords {pixel_coords}")
                print(f"World coordinates: {world_coords}")
                
                # Perform pickup
                success = self.pickup_bottle(world_coords)
                if success:
                    print("Bottle pickup successful!")
                else:
                    print("Bottle pickup failed!")
            
            # Auto-pickup mode (optional)
            # Uncomment the following lines for automatic pickup
            # elif bottles and len(bottles) == 1:
            #     # Auto pickup if only one bottle is detected
            #     time.sleep(2)  # Wait 2 seconds to confirm detection
            #     # ... pickup code here
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        self.move_to_home()
        print("Detection system stopped")

def main():
    """Main function"""
    try:
        robot = BottlePickupRobot()
        robot.run_detection_loop()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()