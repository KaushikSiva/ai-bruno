#!/usr/bin/env python3
"""
Bruno Explore Vision Enhanced - Movement with GPT Vision and Advanced Obstacle Avoidance
Bruno moves around, stops every 5 seconds to take a picture,
gets description from OpenAI Vision, then continues moving.
Uses real-time obstacle detection and intelligent avoidance strategies.
Press 'x' to stop.
"""

import os
import sys
import time
import base64
import io
import threading
import select
import termios
import tty
import numpy as np
import random
import logging

from PIL import Image

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("Error: OpenCV not available. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Error: OpenAI library not available. Install with: pip install openai")
    sys.exit(1)

try:
    from robot_control.movement_controller import MovementController
    BRUNO_MODULES_AVAILABLE = True
except ImportError:
    print("Warning: Bruno movement controller not available - simulation mode")
    BRUNO_MODULES_AVAILABLE = False

# Import collision avoidance from gpt.py
try:
    from gpt import CollisionAvoidance, ObstacleInfo
    COLLISION_AVOIDANCE_AVAILABLE = True
except ImportError:
    print("Warning: Collision avoidance not available - using basic avoidance")
    COLLISION_AVOIDANCE_AVAILABLE = False

class ObstacleAvoidanceSystem:
    """Advanced obstacle avoidance system for Bruno"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Safety zones (in pixels, assuming 640x480 camera)
        self.danger_zone = self.config.get('danger_zone', 80)      # Emergency stop
        self.caution_zone = self.config.get('caution_zone', 120)   # Slow down and avoid
        self.safe_zone = self.config.get('safe_zone', 180)         # Normal operation
        
        # Movement parameters
        self.normal_speed = self.config.get('normal_speed', 40)
        self.caution_speed = self.config.get('caution_speed', 25)
        self.emergency_speed = 0
        
        # State tracking
        self.last_obstacle_check = 0
        self.obstacle_check_interval = self.config.get('obstacle_check_interval', 0.2)
        self.emergency_stop_active = False
        self.avoidance_start_time = 0
        
        # Stuck detection
        self.movement_history = []
        self.max_history_length = 20
        self.stuck_count = 0
        self.max_stuck_count = 3
        
        self.logger.info("Obstacle avoidance system initialized")
    
    def _default_config(self):
        return {
            'danger_zone': 80,      # pixels
            'caution_zone': 120,    # pixels
            'safe_zone': 180,       # pixels
            'normal_speed': 40,
            'caution_speed': 25,
            'obstacle_check_interval': 0.2,
            'enable_vision_detection': True,
            'enable_stuck_detection': True
        }
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_obstacles_vision(self, frame):
        """Detect obstacles using computer vision"""
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
                distance = self.estimate_distance_simple(w, h, W, H)
                
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
            self.logger.error(f"Vision obstacle detection error: {e}")
        
        return obstacles
    
    def estimate_distance_simple(self, width, height, frame_width, frame_height):
        """Simple distance estimation based on object size"""
        apparent_size = width * height
        max_size = frame_width * frame_height * 0.1
        
        distance_ratio = 1.0 - (apparent_size / max_size)
        distance_ratio = max(0.0, min(1.0, distance_ratio))
        
        return distance_ratio * 300  # Max distance of 300 pixels
    
    def analyze_obstacles(self, obstacles):
        """Analyze obstacles and determine avoidance strategy"""
        if not obstacles:
            return {
                'action': 'continue',
                'direction': 'forward',
                'speed': self.normal_speed,
                'emergency_stop': False,
                'message': 'No obstacles detected'
            }
        
        # Find closest obstacle
        closest_obstacle = min(obstacles, key=lambda x: x['distance'])
        distance = closest_obstacle['distance']
        angle = closest_obstacle['angle']
        
        self.logger.debug(f"Closest obstacle: {distance:.1f}px at {angle:.1f}°")
        
        # Determine action based on distance
        if distance < self.danger_zone:
            return {
                'action': 'emergency_stop',
                'direction': 'stop',
                'speed': self.emergency_speed,
                'emergency_stop': True,
                'message': f'DANGER: Obstacle at {distance:.1f}px'
            }
        elif distance < self.caution_zone:
            return self.plan_avoidance(closest_obstacle)
        else:  # Safe zone
            return {
                'action': 'continue',
                'direction': 'forward',
                'speed': self.normal_speed,
                'emergency_stop': False,
                'message': f'Safe: Obstacle at {distance:.1f}px'
            }
    
    def plan_avoidance(self, obstacle):
        """Plan avoidance strategy based on obstacle position"""
        angle = obstacle['angle']
        
        if abs(angle) < 20:  # Obstacle directly ahead
            # Choose left or right based on which side has more space
            if angle > 0:  # Slightly to the right
                return {
                    'action': 'avoid',
                    'direction': 'arc_left',
                    'speed': self.caution_speed,
                    'emergency_stop': False,
                    'message': f'AVOID: Obstacle ahead, turning left'
                }
            else:  # Slightly to the left
                return {
                    'action': 'avoid',
                    'direction': 'arc_right',
                    'speed': self.caution_speed,
                    'emergency_stop': False,
                    'message': f'AVOID: Obstacle ahead, turning right'
                }
        elif angle > 20:  # Obstacle to the right
            return {
                'action': 'avoid',
                'direction': 'arc_left',
                'speed': self.caution_speed,
                'emergency_stop': False,
                'message': f'AVOID: Obstacle to right, turning left'
            }
        else:  # Obstacle to the left
            return {
                'action': 'avoid',
                'direction': 'arc_right',
                'speed': self.caution_speed,
                'emergency_stop': False,
                'message': f'AVOID: Obstacle to left, turning right'
            }
    
    def detect_stuck(self, is_moving):
        """Detect if robot is stuck based on movement patterns"""
        current_time = time.time()
        
        # Add current movement to history
        if is_moving:
            self.movement_history.append(current_time)
            if len(self.movement_history) > self.max_history_length:
                self.movement_history.pop(0)
        
        # Check if we've been trying to move but not making progress
        if (len(self.movement_history) >= self.max_history_length and
            current_time - self.avoidance_start_time > 5.0):
            
            self.stuck_count += 1
            if self.stuck_count >= self.max_stuck_count:
                self.stuck_count = 0
                return True
        
        return False
    
    def get_unstuck_strategy(self):
        """Get strategy to get unstuck"""
        strategies = [
            {'direction': 'backward', 'duration': 1.0, 'speed': 30, 'message': 'Backing up'},
            {'direction': 'left', 'duration': 2.0, 'speed': 35, 'message': 'Turning left'},
            {'direction': 'right', 'duration': 2.0, 'speed': 35, 'message': 'Turning right'},
            {'direction': 'arc_left', 'duration': 1.5, 'speed': 25, 'message': 'Arc turning left'},
            {'direction': 'arc_right', 'duration': 1.5, 'speed': 25, 'message': 'Arc turning right'}
        ]
        return random.choice(strategies)

class BrunoExplorerEnhanced:
    def __init__(self):
        self.setup_openai()
        self.setup_camera()
        self.setup_movement()
        self.setup_obstacle_avoidance()
        
        # Control variables
        self.running = False
        self.last_photo_time = 0
        self.photo_interval = 5  # seconds
        
        # Navigation state
        self.navigation_state = "exploring"  # exploring, avoiding, stuck, emergency_stop
        self.current_direction = "forward"
        self.is_moving = False
        
        print("🤖 Bruno Explorer Enhanced initialized with advanced obstacle avoidance")
        print("Press 'x' to stop the robot")
    
    def setup_openai(self):
        """Setup OpenAI client"""
        if not os.environ.get("OPENAI_API_KEY"):
            print("✗ OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        self.client = OpenAI()
    
    def setup_camera(self):
        """Setup camera capture"""
        try:
            # Try multiple camera sources
            camera_sources = [
                'http://127.0.0.1:8080?action=stream',
                'http://localhost:8080?action=stream',
                0, 1, 2
            ]
            
            self.cap = None
            for source in camera_sources:
                print(f"Trying camera source: {source}")
                cap = cv2.VideoCapture(source)
                
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    ret, frame = cap.read()
                    if ret:
                        print(f"✓ Camera connected: {source}")
                        self.cap = cap
                        break
                    else:
                        cap.release()
            
            if not self.cap:
                print("✗ No working camera found")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error setting up camera: {e}")
            sys.exit(1)
    
    def setup_movement(self):
        """Setup movement controller"""
        if BRUNO_MODULES_AVAILABLE:
            try:
                config = {
                    'max_speed': 60,  # Medium speed
                    'min_speed': 20,
                    'turn_multiplier': 0.8,
                    'movement_timeout': 0.5
                }
                self.movement = MovementController(config)
                self.hardware_available = True
                print("✓ Movement controller initialized")
            except Exception as e:
                print(f"Warning: Hardware not available - {e}")
                self.movement = None
                self.hardware_available = False
        else:
            self.movement = None
            self.hardware_available = False
    
    def setup_obstacle_avoidance(self):
        """Setup obstacle avoidance system"""
        config = {
            'danger_zone': 80,
            'caution_zone': 120,
            'safe_zone': 180,
            'normal_speed': 40,
            'caution_speed': 25,
            'obstacle_check_interval': 0.2,
            'enable_vision_detection': True,
            'enable_stuck_detection': True
        }
        self.obstacle_system = ObstacleAvoidanceSystem(config)
        print("✓ Advanced obstacle avoidance system initialized")
    
    def capture_frame(self):
        """Capture current frame from camera"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def capture_image(self):
        """Capture image from camera"""
        if not self.cap:
            return None
        
        print("📸 Capturing image...")
        
        # Capture a few frames to get a good one
        for i in range(3):
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                # Save image with timestamp
                timestamp = time.strftime("%H%M%S")
                filename = f"bruno_explore_{timestamp}.jpg"
                image.save(filename)
                print(f"✓ Image saved: {filename}")
                return image
        
        print("✗ Failed to capture image")
        return None
    
    def ask_gpt_vision(self, image):
        """Ask GPT Vision to describe the image"""
        try:
            # Encode image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            print("🤖 Asking GPT Vision...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe what you see in this image. Focus on: bottles, bins/trash cans, obstacles, walls, furniture, people, and the general environment. Keep it concise."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data,
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except APIError as e:
            print(f"✗ OpenAI API error: {e}")
            return None
        except Exception as e:
            print(f"✗ Vision error: {e}")
            return None
    
    def move_robot(self, direction, duration=0.5):
        """Move robot in specified direction"""
        if not self.hardware_available or not self.movement:
            print(f"[SIM] Moving {direction} for {duration}s")
            self.is_moving = True
            time.sleep(duration)
            self.is_moving = False
            return
        
        try:
            self.is_moving = True
            
            if direction == "forward":
                self.movement.move_forward(speed=40)
            elif direction == "backward":
                self.movement.move_backward(speed=30)
            elif direction == "left":
                self.movement.turn('LEFT', speed=35)
            elif direction == "right":
                self.movement.turn('RIGHT', speed=35)
            elif direction == "arc_left":
                self.movement.move_forward(speed=30)
                time.sleep(0.1)
                self.movement.turn('LEFT', speed=20)
            elif direction == "arc_right":
                self.movement.move_forward(speed=30)
                time.sleep(0.1)
                self.movement.turn('RIGHT', speed=20)
            elif direction == "stop":
                self.movement.stop()
                self.is_moving = False
                return
            
            time.sleep(duration)
            self.movement.stop()
            self.is_moving = False
            
        except Exception as e:
            print(f"Movement error: {e}")
            self.movement.stop()
            self.is_moving = False
    
    def stop_robot(self):
        """Stop robot movement"""
        if self.hardware_available and self.movement:
            self.movement.stop()
        else:
            print("[SIM] Robot stopped")
        self.is_moving = False
    
    def check_for_stop_key(self):
        """Check if 'x' key is pressed (non-blocking)"""
        try:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char.lower() == 'x':
                    return True
        except:
            pass
        return False
    
    def run(self):
        """Main exploration loop with enhanced obstacle avoidance"""
        print("\n" + "=" * 60)
        print("🚀 BRUNO ENHANCED EXPLORATION MODE STARTING")
        print("=" * 60)
        print("• Takes photo every 5 seconds")
        print("• Gets GPT Vision description")  
        print("• Real-time obstacle detection and avoidance")
        print("• Intelligent navigation and stuck detection")
        print("• Press 'x' to stop")
        print("=" * 60)
        
        # Setup terminal for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            self.running = True
            self.last_photo_time = time.time()
            self.obstacle_system.avoidance_start_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Check for stop key
                if self.check_for_stop_key():
                    print("\n🛑 Stop key pressed - shutting down")
                    break
                
                # Take photo every 5 seconds
                if current_time - self.last_photo_time >= self.photo_interval:
                    print(f"\n⏰ Photo time! ({self.photo_interval}s interval)")
                    
                    # Stop robot for photo
                    self.stop_robot()
                    time.sleep(0.5)  # Brief pause
                    
                    # Capture and analyze image
                    image = self.capture_image()
                    if image:
                        description = self.ask_gpt_vision(image)
                        if description:
                            print("\n" + "=" * 50)
                            print("🔍 GPT VISION DESCRIPTION")
                            print("=" * 50)
                            print(description)
                            print("=" * 50)
                        else:
                            print("✗ No description received")
                    
                    self.last_photo_time = current_time
                    print("🚶 Resuming movement...\n")
                
                # Real-time obstacle detection and avoidance
                frame = self.capture_frame()
                if frame is not None:
                    obstacles = self.obstacle_system.detect_obstacles_vision(frame)
                    avoidance_plan = self.obstacle_system.analyze_obstacles(obstacles)
                    
                    # Execute avoidance plan
                    if avoidance_plan['emergency_stop']:
                        print(f"⚠️  {avoidance_plan['message']}")
                        self.stop_robot()
                        self.navigation_state = "emergency_stop"
                        self.obstacle_system.emergency_stop_active = True
                        self.obstacle_system.avoidance_start_time = current_time
                        time.sleep(1.0)  # Wait before resuming
                        self.obstacle_system.emergency_stop_active = False
                        self.navigation_state = "exploring"
                        
                    elif avoidance_plan['action'] == 'avoid':
                        print(f"🔄 {avoidance_plan['message']}")
                        self.move_robot(avoidance_plan['direction'], duration=0.3)
                        self.navigation_state = "avoiding"
                        self.current_direction = avoidance_plan['direction']
                        
                    elif avoidance_plan['action'] == 'continue':
                        if self.navigation_state != "exploring":
                            print(f"✅ {avoidance_plan['message']}")
                            self.navigation_state = "exploring"
                        
                        # Continue with current direction or choose new one
                        if current_time - self.obstacle_system.last_obstacle_check > 2.0:
                            # Change direction occasionally to explore
                            directions = ["forward", "arc_left", "arc_right"]
                            self.current_direction = random.choice(directions)
                            self.obstacle_system.last_obstacle_check = current_time
                            print(f"🔄 Exploration direction change: {self.current_direction}")
                        
                        self.move_robot(self.current_direction, duration=0.3)
                
                # Stuck detection
                if self.obstacle_system.detect_stuck(self.is_moving):
                    print("🤖 Robot appears to be stuck! Attempting recovery...")
                    unstuck_strategy = self.obstacle_system.get_unstuck_strategy()
                    print(f"🔄 {unstuck_strategy['message']}")
                    self.move_robot(unstuck_strategy['direction'], duration=unstuck_strategy['duration'])
                    self.obstacle_system.avoidance_start_time = current_time
                    self.navigation_state = "exploring"
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by Ctrl+C")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.stop_robot()
        
        if self.cap:
            self.cap.release()
            print("Camera released")
        
        print("✅ Bruno Explorer Enhanced cleanup complete")

def main():
    """Main function"""
    try:
        explorer = BrunoExplorerEnhanced()
        explorer.run()
    except Exception as e:
        print(f"Failed to start Bruno Explorer Enhanced: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
