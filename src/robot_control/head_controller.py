#!/usr/bin/env python3
"""
Head Controller for MasterPi Robot (Bruno)
Controls head movements including nodding when bottles are detected
"""

import time
import threading
import logging
from typing import Optional, Callable
from common.ros_robot_controller_sdk import Board

class HeadController:
    def __init__(self, config: dict = None):
        self.board = Board()
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Head servo configuration
        self.head_servo_id = self.config['head_servo']['id']
        self.positions = self.config['head_servo']['positions']
        self.timings = self.config['head_servo']['timings']
        
        # Control state
        self.is_nodding = False
        self.nod_thread = None
        self.stop_nodding = threading.Event()
        
        # Initialize head to center position
        self.center_head()
        
    def _default_config(self) -> dict:
        """Default head controller configuration"""
        return {
            'head_servo': {
                'id': 2,  # Default servo ID - may need adjustment
                'positions': {
                    'center': 1500,
                    'left': 1200,
                    'right': 1800,
                    'up': 1300,
                    'down': 1700
                },
                'timings': {
                    'normal_movement': 0.5,
                    'slow_movement': 1.0,
                    'fast_movement': 0.3,
                    'nod_pause': 0.8
                }
            },
            'nodding_patterns': {
                'excited': {
                    'sequence': ['left', 'right', 'left', 'right'],
                    'speed': 'fast_movement',
                    'repetitions': 3
                },
                'acknowledgment': {
                    'sequence': ['up', 'down', 'center'],
                    'speed': 'normal_movement',
                    'repetitions': 2
                },
                'scanning': {
                    'sequence': ['left', 'center', 'right', 'center'],
                    'speed': 'slow_movement',
                    'repetitions': 1
                }
            }
        }
    
    def setup_logging(self):
        """Setup logging for the head controller"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def move_to_position(self, position: str, duration: float = None):
        """Move head to a specific position"""
        if position not in self.positions:
            self.logger.warning(f"Unknown position: {position}")
            return False
        
        if duration is None:
            duration = self.timings['normal_movement']
        
        try:
            servo_position = self.positions[position]
            self.board.pwm_servo_set_position(duration, [[self.head_servo_id, servo_position]])
            self.logger.debug(f"Moving head to {position} (position: {servo_position})")
            return True
        except Exception as e:
            self.logger.error(f"Error moving head to {position}: {e}")
            return False
    
    def center_head(self):
        """Move head to center position"""
        self.move_to_position('center')
        time.sleep(self.timings['normal_movement'])
    
    def nod_yes(self, repetitions: int = 2):
        """Nod head up and down (yes gesture)"""
        self.logger.info(f"Nodding yes {repetitions} times")
        
        for _ in range(repetitions):
            self.move_to_position('up', self.timings['fast_movement'])
            time.sleep(self.timings['fast_movement'] + 0.1)
            
            self.move_to_position('down', self.timings['fast_movement'])
            time.sleep(self.timings['fast_movement'] + 0.1)
        
        self.center_head()
    
    def nod_no(self, repetitions: int = 2):
        """Shake head left and right (no gesture)"""
        self.logger.info(f"Shaking head no {repetitions} times")
        
        for _ in range(repetitions):
            self.move_to_position('left', self.timings['fast_movement'])
            time.sleep(self.timings['fast_movement'] + 0.1)
            
            self.move_to_position('right', self.timings['fast_movement'])
            time.sleep(self.timings['fast_movement'] + 0.1)
        
        self.center_head()
    
    def execute_pattern(self, pattern_name: str):
        """Execute a predefined nodding pattern"""
        if pattern_name not in self.config['nodding_patterns']:
            self.logger.warning(f"Unknown pattern: {pattern_name}")
            return
        
        pattern = self.config['nodding_patterns'][pattern_name]
        sequence = pattern['sequence']
        speed_key = pattern['speed']
        repetitions = pattern['repetitions']
        
        movement_time = self.timings[speed_key]
        
        self.logger.info(f"Executing pattern: {pattern_name}")
        
        for rep in range(repetitions):
            for position in sequence:
                if self.stop_nodding.is_set():
                    break
                    
                self.move_to_position(position, movement_time)
                time.sleep(movement_time + 0.1)
            
            if rep < repetitions - 1:  # Pause between repetitions
                time.sleep(self.timings['nod_pause'])
        
        self.center_head()
    
    def bottle_detected_response(self, bottle_count: int, confidence: float):
        """Respond to bottle detection with appropriate head movement"""
        if bottle_count == 0:
            return
        
        if confidence > 0.8:
            # High confidence - excited nodding
            self.execute_pattern('excited')
        elif confidence > 0.6:
            # Medium confidence - acknowledgment
            self.execute_pattern('acknowledgment') 
        else:
            # Low confidence - scanning
            self.execute_pattern('scanning')
    
    def look_at_position(self, pixel_x: int, pixel_y: int, frame_width: int = 640, frame_height: int = 480):
        """Move head to look at a specific position in the camera frame"""
        # Convert pixel position to servo position
        center_x = frame_width // 2
        
        # Calculate offset from center (-1 to 1)
        offset_x = (pixel_x - center_x) / center_x
        
        # Convert to servo positions
        center_pos = self.positions['center']
        left_pos = self.positions['left']
        right_pos = self.positions['right']
        
        # Calculate horizontal position
        if offset_x < 0:  # Look left
            target_pos = center_pos + int((center_pos - left_pos) * abs(offset_x))
        else:  # Look right
            target_pos = center_pos + int((right_pos - center_pos) * offset_x)
        
        # Clamp to valid range
        target_pos = max(left_pos, min(right_pos, target_pos))
        
        try:
            self.board.pwm_servo_set_position(self.timings['normal_movement'], 
                                            [[self.head_servo_id, target_pos]])
            self.logger.debug(f"Looking at pixel ({pixel_x}, {pixel_y}) -> servo position {target_pos}")
        except Exception as e:
            self.logger.error(f"Error looking at position: {e}")
    
    def calibrate_servo(self):
        """Calibration routine to test head servo positions"""
        self.logger.info("Starting head servo calibration")
        
        positions_to_test = ['center', 'left', 'right', 'up', 'down']
        
        for position in positions_to_test:
            self.logger.info(f"Testing position: {position}")
            self.move_to_position(position)
            time.sleep(2)
        
        self.center_head()
        self.logger.info("Calibration complete")
    
    def cleanup(self):
        """Cleanup resources and stop any ongoing operations"""
        self.stop_continuous_nodding()
        self.center_head()
        self.logger.info("Head controller cleanup complete")
    
    def stop_continuous_nodding(self):
        """Stop continuous nodding (placeholder for compatibility)"""
        if self.is_nodding:
            self.stop_nodding.set()
            self.is_nodding = False