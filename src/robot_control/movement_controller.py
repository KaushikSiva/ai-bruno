#!/usr/bin/env python3
"""
Movement Controller for MasterPi Robot
Controls robot chassis movement for bottle approach
"""

import time
import logging
from typing import Dict, Optional
import common.mecanum as mecanum
from common.ros_robot_controller_sdk import Board

class MovementController:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Initialize hardware
        try:
            self.chassis = mecanum.MecanumChassis()
            self.board = Board()
        except Exception as e:
            self.logger.error(f"Failed to initialize movement hardware: {e}")
            self.chassis = None
            self.board = None
        
        # Movement state
        self.is_moving = False
        self.current_command = None
        self.last_movement_time = 0
        
        # Safety parameters
        self.max_speed = self.config.get('max_speed', 50)
        self.min_speed = self.config.get('min_speed', 15)
        self.movement_timeout = self.config.get('movement_timeout', 0.5)  # seconds
        
        self.logger.info("Movement controller initialized")
    
    def _default_config(self) -> Dict:
        """Default movement configuration"""
        return {
            'max_speed': 50,        # Maximum movement speed (0-100)
            'min_speed': 15,        # Minimum movement speed
            'turn_multiplier': 0.8, # Turn speed multiplier
            'movement_timeout': 0.5, # Movement command timeout
            'safety_enabled': True,  # Enable safety checks
            'smooth_acceleration': True  # Smooth speed changes
        }
    
    def setup_logging(self):
        """Setup logging for movement controller"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def stop(self):
        """Stop all robot movement"""
        if not self.chassis:
            return
            
        try:
            self.chassis.set_velocity(0, 0, 0)
            self.is_moving = False
            self.current_command = None
            self.logger.debug("Robot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping robot: {e}")
    
    def move_forward(self, speed: float):
        """Move robot forward at specified speed"""
        if not self.chassis:
            self.logger.warning("Chassis not available - simulating movement")
            return
            
        speed = max(self.min_speed, min(speed, self.max_speed))
        
        try:
            # Move forward (90 degrees in mecanum coordinate system)
            self.chassis.set_velocity(speed, 90, 0)
            self.is_moving = True
            self.last_movement_time = time.time()
            self.logger.debug(f"Moving forward at speed {speed}")
        except Exception as e:
            self.logger.error(f"Error moving forward: {e}")
            self.stop()
    
    def turn(self, direction: str, speed: float):
        """Turn robot in specified direction"""
        if not self.chassis:
            self.logger.warning("Chassis not available - simulating turn")
            return
            
        speed = max(self.min_speed, min(speed, self.max_speed))
        turn_speed = speed * self.config.get('turn_multiplier', 0.8)
        
        try:
            if direction.upper() == 'LEFT':
                # Turn left (negative rotation)
                self.chassis.set_velocity(0, 0, -turn_speed)
            elif direction.upper() == 'RIGHT':
                # Turn right (positive rotation)
                self.chassis.set_velocity(0, 0, turn_speed)
            else:
                self.logger.warning(f"Unknown turn direction: {direction}")
                return
            
            self.is_moving = True
            self.last_movement_time = time.time()
            self.logger.debug(f"Turning {direction} at speed {turn_speed}")
        except Exception as e:
            self.logger.error(f"Error turning {direction}: {e}")
            self.stop()
    
    def move_and_turn(self, forward_speed: float, turn_direction: str, turn_speed: float):
        """Move forward while turning (combined movement)"""
        if not self.chassis:
            self.logger.warning("Chassis not available - simulating combined movement")
            return
            
        forward_speed = max(0, min(forward_speed, self.max_speed))
        turn_speed = max(0, min(turn_speed, self.max_speed))
        
        # Apply turn multiplier
        turn_speed = turn_speed * self.config.get('turn_multiplier', 0.8)
        
        try:
            # Determine rotation direction
            if turn_direction.upper() == 'LEFT':
                rotation = -turn_speed
            elif turn_direction.upper() == 'RIGHT':
                rotation = turn_speed
            else:
                rotation = 0
            
            # Combined movement: forward + rotation
            if forward_speed > 0:
                self.chassis.set_velocity(forward_speed, 90, rotation)
            else:
                # Pure rotation if no forward speed
                self.chassis.set_velocity(0, 0, rotation)
            
            self.is_moving = True
            self.last_movement_time = time.time()
            self.logger.debug(f"Moving forward {forward_speed}, turning {turn_direction} {turn_speed}")
            
        except Exception as e:
            self.logger.error(f"Error in combined movement: {e}")
            self.stop()
    
    def execute_movement_command(self, command: Dict):
        """Execute a movement command from distance estimator"""
        if not command:
            self.stop()
            return
        
        action = command.get('action', 'STOP')
        forward_speed = command.get('forward_speed', 0)
        turn_direction = command.get('turn_direction', 'STRAIGHT')
        turn_speed = command.get('turn_speed', 0)
        
        self.current_command = command.copy()
        
        if action == 'STOP':
            self.stop()
            self.logger.info("STOP: Bottle reached!")
            
        elif action in ['APPROACH_SLOW', 'APPROACH_NORMAL']:
            if turn_direction == 'STRAIGHT':
                # Move straight forward
                self.move_forward(forward_speed)
                self.logger.info(f"{action}: Moving forward at {forward_speed}%")
            else:
                # Move forward while turning
                self.move_and_turn(forward_speed, turn_direction, turn_speed)
                self.logger.info(f"{action}: Forward {forward_speed}%, turn {turn_direction} {turn_speed}%")
        
        else:
            self.logger.warning(f"Unknown action: {action}")
            self.stop()
    
    def check_movement_timeout(self):
        """Check if movement command has timed out (safety feature)"""
        if self.is_moving and self.last_movement_time > 0:
            elapsed = time.time() - self.last_movement_time
            if elapsed > self.movement_timeout:
                self.logger.warning("Movement timeout - stopping for safety")
                self.stop()
                return True
        return False
    
    def emergency_stop(self):
        """Emergency stop - immediately halt all movement"""
        self.logger.warning("EMERGENCY STOP")
        self.stop()
    
    def get_status(self) -> Dict:
        """Get current movement status"""
        return {
            'is_moving': self.is_moving,
            'current_command': self.current_command,
            'last_movement_time': self.last_movement_time,
            'time_since_last_movement': time.time() - self.last_movement_time if self.last_movement_time > 0 else 0
        }
    
    def calibrate_speeds(self):
        """Calibration routine to test different movement speeds"""
        self.logger.info("Starting movement calibration...")
        
        if not self.chassis:
            self.logger.warning("Chassis not available - calibration skipped")
            return
        
        speeds_to_test = [15, 25, 35, 50]
        
        for speed in speeds_to_test:
            self.logger.info(f"Testing forward speed: {speed}")
            self.move_forward(speed)
            time.sleep(2)
            self.stop()
            time.sleep(1)
        
        # Test turning
        for direction in ['LEFT', 'RIGHT']:
            for speed in [20, 35]:
                self.logger.info(f"Testing turn {direction} speed: {speed}")
                self.turn(direction, speed)
                time.sleep(2)
                self.stop()
                time.sleep(1)
        
        self.logger.info("Movement calibration complete")
    
    def cleanup(self):
        """Cleanup movement controller"""
        self.stop()
        self.logger.info("Movement controller cleanup complete")