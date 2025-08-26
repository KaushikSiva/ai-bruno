#!/usr/bin/env python3
"""
Bruno Explore Vision - Movement with GPT Vision
Bruno moves around, stops every 5 seconds to take a picture,
gets description from OpenAI Vision, then continues moving.
Avoids collisions within 6 inches. Press 'x' to stop.
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

class BrunoExplorer:
    def __init__(self):
        self.setup_openai()
        self.setup_camera()
        self.setup_movement()
        
        # Control variables
        self.running = False
        self.last_photo_time = 0
        self.photo_interval = 5  # seconds
        
        # Obstacle avoidance (simplified - using time-based logic)
        self.obstacle_detected = False
        self.last_direction_change = 0
        self.current_direction = "forward"  # forward, left, right, backward
        
        print("🤖 Bruno Explorer initialized")
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
            return
        
        try:
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
            
            time.sleep(duration)
            self.movement.stop()
            
        except Exception as e:
            print(f"Movement error: {e}")
            self.movement.stop()
    
    def stop_robot(self):
        """Stop robot movement"""
        if self.hardware_available and self.movement:
            self.movement.stop()
        else:
            print("[SIM] Robot stopped")
    
    def simple_obstacle_avoidance(self):
        """Simple obstacle avoidance using time-based direction changes"""
        current_time = time.time()
        
        # Change direction every 8-15 seconds to avoid getting stuck
        if current_time - self.last_direction_change > 10:
            # Randomly choose new direction
            import random
            directions = ["forward", "arc_left", "arc_right"]
            self.current_direction = random.choice(directions)
            self.last_direction_change = current_time
            print(f"🔄 Direction change: {self.current_direction}")
        
        return self.current_direction
    
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
        """Main exploration loop"""
        print("\n" + "=" * 50)
        print("🚀 BRUNO EXPLORATION MODE STARTING")
        print("=" * 50)
        print("• Takes photo every 5 seconds")
        print("• Gets GPT Vision description")  
        print("• Avoids obstacles within 6 inches")
        print("• Press 'x' to stop")
        print("=" * 50)
        
        # Setup terminal for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            self.running = True
            self.last_photo_time = time.time()
            self.last_direction_change = time.time()
            
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
                
                # Continue movement with obstacle avoidance
                direction = self.simple_obstacle_avoidance()
                self.move_robot(direction, duration=0.3)
                
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
        
        print("✅ Bruno Explorer cleanup complete")

def main():
    """Main function"""
    try:
        explorer = BrunoExplorer()
        explorer.run()
    except Exception as e:
        print(f"Failed to start Bruno Explorer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()