#!/usr/bin/env python3
"""
Roomba-style Navigation for MasterPi Robot
Implements autonomous movement patterns with obstacle avoidance
"""

import time
import random
import logging
import math
from typing import Dict, Tuple, List, Optional
from enum import Enum

try:
    import common.mecanum as mecanum
    from common.ros_robot_controller_sdk import Board
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

class NavigationState(Enum):
    IDLE = "idle"
    FORWARD = "forward"
    TURNING = "turning"
    BACKING_UP = "backing_up"
    SPIRAL = "spiral"
    WALL_FOLLOW = "wall_follow"
    BOTTLE_APPROACH = "bottle_approach"

class RoombaNavigator:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Initialize hardware
        self.hardware_available = HARDWARE_AVAILABLE
        if self.hardware_available:
            try:
                self.chassis = mecanum.MecanumChassis()
                self.board = Board()
                self.logger.info("✅ Movement hardware initialized")
            except Exception as e:
                self.logger.warning(f"⚠️  Hardware initialization failed: {e}")
                self.hardware_available = False
        else:
            self.logger.warning("⚠️  Hardware not available - simulation mode")
        
        # Navigation state
        self.current_state = NavigationState.IDLE
        self.state_start_time = 0
        self.last_movement_time = 0
        self.is_moving = False
        
        # Movement parameters
        self.forward_speed = self.config.get('forward_speed', 30)
        self.turn_speed = self.config.get('turn_speed', 25)
        self.backup_speed = self.config.get('backup_speed', 20)
        
        # Navigation patterns
        self.spiral_angle = 0
        self.wall_follow_side = "left"  # or "right"
        self.random_turn_direction = "left"
        
        # Obstacle detection
        self.obstacle_detected = False
        self.stuck_count = 0
        self.max_stuck_count = 3
        
        # Movement history for stuck detection
        self.position_history = []
        self.max_history_length = 10
        
        self.logger.info("🤖 Roomba Navigator initialized")
    
    def _default_config(self) -> Dict:
        """Default navigation configuration"""
        return {
            'forward_speed': 30,
            'turn_speed': 25,
            'backup_speed': 20,
            'obstacle_threshold': 30,  # cm
            'forward_time_min': 2.0,   # seconds
            'forward_time_max': 8.0,   # seconds
            'turn_time_min': 1.0,      # seconds
            'turn_time_max': 3.0,      # seconds
            'backup_time': 1.5,        # seconds
            'spiral_time': 10.0,       # seconds
            'patterns': ['forward', 'spiral', 'random_turn'],
            'pattern_change_interval': 30.0,  # seconds
            'stuck_detection': True,
            'wall_following': True
        }
    
    def setup_logging(self):
        """Setup logging for navigator"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def execute_movement(self, velocity: float, direction: float, rotation: float):
        """Execute movement command"""
        if self.hardware_available:
            try:
                self.chassis.set_velocity(velocity, direction, rotation)
                self.is_moving = velocity > 0 or abs(rotation) > 0
                self.last_movement_time = time.time()
            except Exception as e:
                self.logger.error(f"Movement execution failed: {e}")
        else:
            # Simulate movement
            self.is_moving = velocity > 0 or abs(rotation) > 0
            self.last_movement_time = time.time()
            if self.is_moving:
                self.logger.info(f"🎯 [SIM] Velocity: {velocity}, Direction: {direction}°, Rotation: {rotation}")
    
    def stop(self):
        """Stop all movement"""
        self.execute_movement(0, 0, 0)
        self.is_moving = False
        self.current_state = NavigationState.IDLE
    
    def move_forward(self, speed: float = None):
        """Move forward at specified speed"""
        speed = speed or self.forward_speed
        self.execute_movement(speed, 90, 0)  # 90° = forward in mecanum system
        self.logger.debug(f"Moving forward at speed {speed}")
    
    def turn_left(self, speed: float = None):
        """Turn left in place"""
        speed = speed or self.turn_speed
        self.execute_movement(0, 0, -speed)  # Negative rotation = left
        self.logger.debug(f"Turning left at speed {speed}")
    
    def turn_right(self, speed: float = None):
        """Turn right in place"""
        speed = speed or self.turn_speed
        self.execute_movement(0, 0, speed)  # Positive rotation = right
        self.logger.debug(f"Turning right at speed {speed}")
    
    def move_backward(self, speed: float = None):
        """Move backward"""
        speed = speed or self.backup_speed
        self.execute_movement(speed, 270, 0)  # 270° = backward
        self.logger.debug(f"Moving backward at speed {speed}")
    
    def arc_turn(self, forward_speed: float, turn_speed: float, direction: str):
        """Execute an arc turn (forward movement with rotation)"""
        rotation = turn_speed if direction == "right" else -turn_speed
        self.execute_movement(forward_speed, 90, rotation)
        self.logger.debug(f"Arc turning {direction}: forward {forward_speed}, rotation {rotation}")
    
    def detect_obstacle(self, distance_data: Optional[Dict] = None) -> bool:
        """
        Detect obstacles using available sensors
        For now, uses simple time-based stuck detection
        """
        if distance_data:
            # If actual distance sensors are available
            min_distance = min(distance_data.values()) if distance_data else float('inf')
            return min_distance < self.config['obstacle_threshold']
        
        # Simple stuck detection based on movement history
        if self.config['stuck_detection']:
            return self.detect_stuck()
        
        return False
    
    def detect_stuck(self) -> bool:
        """Detect if robot is stuck based on movement patterns"""
        # This is a simplified stuck detection
        # In a real implementation, you'd use encoders or IMU data
        
        current_time = time.time()
        
        # Add current position to history (simulated)
        if self.is_moving:
            # Simulate position tracking
            self.position_history.append(current_time)
            if len(self.position_history) > self.max_history_length:
                self.position_history.pop(0)
        
        # Check if we've been trying to move but not making progress
        if (len(self.position_history) >= self.max_history_length and
            self.current_state == NavigationState.FORWARD and
            current_time - self.state_start_time > 5.0):
            
            self.stuck_count += 1
            if self.stuck_count >= self.max_stuck_count:
                self.stuck_count = 0
                return True
        
        return False
    
    def choose_random_pattern(self) -> NavigationState:
        """Choose a random movement pattern"""
        patterns = [
            NavigationState.FORWARD,
            NavigationState.SPIRAL,
            NavigationState.TURNING
        ]
        return random.choice(patterns)
    
    def update_navigation(self, bottle_detected: bool = False, bottle_position: Optional[Tuple] = None,
                         distance_data: Optional[Dict] = None) -> Dict:
        """
        Main navigation update function
        Returns movement command information
        """
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        # Check for obstacles
        self.obstacle_detected = self.detect_obstacle(distance_data)
        
        # Priority 1: Bottle approach mode
        if bottle_detected and bottle_position:
            return self._handle_bottle_approach(bottle_position)
        
        # Priority 2: Obstacle avoidance
        if self.obstacle_detected:
            return self._handle_obstacle_avoidance()
        
        # Priority 3: Normal navigation patterns
        return self._handle_normal_navigation(state_duration)
    
    def _handle_bottle_approach(self, bottle_position: Tuple) -> Dict:
        """Handle movement when bottle is detected"""
        self.current_state = NavigationState.BOTTLE_APPROACH
        
        # Calculate turn direction based on bottle position
        frame_center_x = 320  # Assuming 640px width
        bottle_x = bottle_position[0]
        
        offset = bottle_x - frame_center_x
        
        if abs(offset) < 50:  # Dead zone - move straight
            self.move_forward(20)  # Slow approach
            return {
                'state': 'bottle_approach',
                'action': 'forward',
                'speed': 20
            }
        elif offset > 0:  # Bottle on right
            self.arc_turn(15, 15, "right")
            return {
                'state': 'bottle_approach',
                'action': 'arc_right',
                'speed': 15
            }
        else:  # Bottle on left
            self.arc_turn(15, 15, "left")
            return {
                'state': 'bottle_approach',
                'action': 'arc_left',
                'speed': 15
            }
    
    def _handle_obstacle_avoidance(self) -> Dict:
        """Handle obstacle avoidance"""
        if self.current_state != NavigationState.BACKING_UP:
            self.current_state = NavigationState.BACKING_UP
            self.state_start_time = time.time()
            self.logger.info("🚫 Obstacle detected - backing up")
        
        state_duration = time.time() - self.state_start_time
        
        if state_duration < self.config['backup_time']:
            self.move_backward()
            return {
                'state': 'backing_up',
                'action': 'backward',
                'duration': state_duration
            }
        else:
            # After backing up, turn random direction
            self.current_state = NavigationState.TURNING
            self.state_start_time = time.time()
            self.random_turn_direction = random.choice(["left", "right"])
            
            if self.random_turn_direction == "left":
                self.turn_left()
            else:
                self.turn_right()
            
            self.logger.info(f"🔄 Turning {self.random_turn_direction} after obstacle")
            
            return {
                'state': 'turning',
                'action': f'turn_{self.random_turn_direction}',
                'duration': 0
            }
    
    def _handle_normal_navigation(self, state_duration: float) -> Dict:
        """Handle normal navigation patterns"""
        
        # State machine for navigation patterns
        if self.current_state == NavigationState.IDLE:
            self._start_new_pattern()
        
        elif self.current_state == NavigationState.FORWARD:
            return self._handle_forward_movement(state_duration)
        
        elif self.current_state == NavigationState.TURNING:
            return self._handle_turning(state_duration)
        
        elif self.current_state == NavigationState.SPIRAL:
            return self._handle_spiral_pattern(state_duration)
        
        elif self.current_state == NavigationState.WALL_FOLLOW:
            return self._handle_wall_following(state_duration)
        
        return {'state': 'idle', 'action': 'none'}
    
    def _start_new_pattern(self):
        """Start a new navigation pattern"""
        self.current_state = self.choose_random_pattern()
        self.state_start_time = time.time()
        self.logger.info(f"🎯 Starting pattern: {self.current_state.value}")
    
    def _handle_forward_movement(self, state_duration: float) -> Dict:
        """Handle forward movement pattern"""
        max_forward_time = random.uniform(
            self.config['forward_time_min'],
            self.config['forward_time_max']
        )
        
        if state_duration < max_forward_time:
            self.move_forward()
            return {
                'state': 'forward',
                'action': 'forward',
                'duration': state_duration,
                'max_duration': max_forward_time
            }
        else:
            # Switch to turning
            self.current_state = NavigationState.TURNING
            self.state_start_time = time.time()
            self.random_turn_direction = random.choice(["left", "right"])
            return self._handle_turning(0)
    
    def _handle_turning(self, state_duration: float) -> Dict:
        """Handle turning pattern"""
        max_turn_time = random.uniform(
            self.config['turn_time_min'],
            self.config['turn_time_max']
        )
        
        if state_duration < max_turn_time:
            if self.random_turn_direction == "left":
                self.turn_left()
            else:
                self.turn_right()
            
            return {
                'state': 'turning',
                'action': f'turn_{self.random_turn_direction}',
                'duration': state_duration,
                'max_duration': max_turn_time
            }
        else:
            # Switch back to forward or new pattern
            self.current_state = NavigationState.FORWARD
            self.state_start_time = time.time()
            return self._handle_forward_movement(0)
    
    def _handle_spiral_pattern(self, state_duration: float) -> Dict:
        """Handle spiral movement pattern"""
        if state_duration < self.config['spiral_time']:
            # Gradually increase turn radius
            radius_factor = 1 + (state_duration / self.config['spiral_time']) * 2
            turn_speed = self.turn_speed / radius_factor
            
            self.arc_turn(self.forward_speed * 0.7, turn_speed, "right")
            
            return {
                'state': 'spiral',
                'action': 'spiral_right',
                'duration': state_duration,
                'radius_factor': radius_factor
            }
        else:
            # Switch to forward movement
            self.current_state = NavigationState.FORWARD
            self.state_start_time = time.time()
            return self._handle_forward_movement(0)
    
    def _handle_wall_following(self, state_duration: float) -> Dict:
        """Handle wall following pattern (simplified)"""
        # This is a basic wall following simulation
        # In real implementation, would use distance sensors
        
        if state_duration < 10.0:  # Follow wall for 10 seconds
            # Simulate wall following by arc turning
            if self.wall_follow_side == "left":
                self.arc_turn(self.forward_speed * 0.8, self.turn_speed * 0.5, "left")
            else:
                self.arc_turn(self.forward_speed * 0.8, self.turn_speed * 0.5, "right")
            
            return {
                'state': 'wall_follow',
                'action': f'follow_{self.wall_follow_side}',
                'duration': state_duration
            }
        else:
            # Switch to different pattern
            self.wall_follow_side = "right" if self.wall_follow_side == "left" else "left"
            self.current_state = NavigationState.FORWARD
            self.state_start_time = time.time()
            return self._handle_forward_movement(0)
    
    def get_status(self) -> Dict:
        """Get current navigation status"""
        return {
            'state': self.current_state.value,
            'is_moving': self.is_moving,
            'obstacle_detected': self.obstacle_detected,
            'stuck_count': self.stuck_count,
            'hardware_available': self.hardware_available,
            'state_duration': time.time() - self.state_start_time
        }
    
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.stop()
        self.logger.warning("🚨 EMERGENCY STOP - Navigation halted")
    
    def cleanup(self):
        """Clean up navigator"""
        self.stop()
        self.logger.info("🏁 Roomba Navigator cleanup complete")