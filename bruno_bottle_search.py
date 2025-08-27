#!/usr/bin/env python3
"""
Bruno Bottle Search - Random Movement with Vision Detection
Bruno moves randomly avoiding obstacles. Every 10 seconds it stops,
takes a picture, and uses GPT Vision to detect plastic bottles.
If bottle found, prints "Done. Plastic bottle found" and stops.
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
import random

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

class BottleSearcher:
    def __init__(self):
        self.setup_openai()
        self.setup_camera()
        self.setup_movement()
        
        # Control variables
        self.running = False
        self.last_check_time = 0
        self.check_interval = 10  # seconds
        self.bottle_found = False
        
        # Movement control
        self.last_direction_change = 0
        self.direction_change_interval = 3  # Change direction every 3 seconds
        self.current_direction = "forward"
        
        print("🔍 Bruno Bottle Searcher initialized")
        print("• Moves randomly avoiding obstacles")
        print("• Checks for bottles every 10 seconds")
        print("• Stops when plastic bottle found")
    
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
                    'max_speed': 50,  # Medium speed for searching
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
        """Capture fresh image from camera"""
        if not self.cap:
            return None
        
        print("📸 Capturing FRESH image for bottle detection...")
        
        # Flush camera buffer by reading multiple frames
        print("🔄 Flushing camera buffer...")
        for i in range(5):
            ret, frame = self.cap.read()
            if ret:
                print(f"   Flushed frame {i+1}/5")
            time.sleep(0.1)  # Small delay between frames
        
        # Now capture the actual image
        print("📷 Capturing actual image...")
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Save image with more precise timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            milliseconds = int(time.time() * 1000) % 1000
            filename = f"bottle_search_{timestamp}_{milliseconds:03d}.jpg"
            image.save(filename)
            print(f"✓ Fresh image saved: {filename}")
            return image
        else:
            print("✗ Failed to capture fresh image")
            return None
    
    def check_for_bottle(self, image):
        """Use GPT Vision to check for plastic bottles"""
        try:
            # Encode image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            print("🤖 Checking for plastic bottles...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Look at this image and determine if there is a PLASTIC BOTTLE visible. Answer with only 'YES' if you see a plastic bottle (water bottle, soda bottle, etc.) or 'NO' if you don't see any plastic bottles. Be specific about plastic bottles only."
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
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            print(f"🔍 GPT Vision result: {result}")
            
            return result == "YES"
            
        except APIError as e:
            print(f"✗ OpenAI API error: {e}")
            return False
        except Exception as e:
            print(f"✗ Vision error: {e}")
            return False
    
    def move_robot(self, direction, duration=0.3):
        """Move robot in specified direction"""
        if not self.hardware_available or not self.movement:
            print(f"[SIM] Moving {direction} for {duration}s")
            return
        
        try:
            if direction == "forward":
                self.movement.move_forward(speed=35)
            elif direction == "backward":
                self.movement.move_backward(speed=25)
            elif direction == "left":
                self.movement.turn('LEFT', speed=30)
            elif direction == "right":
                self.movement.turn('RIGHT', speed=30)
            elif direction == "arc_left":
                self.movement.move_forward(speed=25)
                time.sleep(0.1)
                self.movement.turn('LEFT', speed=20)
            elif direction == "arc_right":
                self.movement.move_forward(speed=25)
                time.sleep(0.1)
                self.movement.turn('RIGHT', speed=20)
            
            time.sleep(duration)
            self.movement.stop()
            
        except Exception as e:
            print(f"Movement error: {e}")
            if self.movement:
                self.movement.stop()
    
    def stop_robot(self):
        """Stop robot movement"""
        if self.hardware_available and self.movement:
            self.movement.stop()
        else:
            print("[SIM] Robot stopped")
    
    def get_random_direction(self):
        """Get random movement direction for exploration"""
        directions = [
            "forward", "forward", "forward",  # More forward movement
            "arc_left", "arc_right", 
            "left", "right"
        ]
        return random.choice(directions)
    
    def random_movement_with_avoidance(self):
        """Random movement with simple obstacle avoidance"""
        current_time = time.time()
        
        # Change direction randomly every few seconds
        if current_time - self.last_direction_change > self.direction_change_interval:
            self.current_direction = self.get_random_direction()
            self.last_direction_change = current_time
            self.direction_change_interval = random.uniform(2, 5)  # Random interval
            print(f"🔄 Direction: {self.current_direction}")
        
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
        """Main bottle search loop"""
        print("\n" + "=" * 50)
        print("🔍 BRUNO BOTTLE SEARCH MODE")
        print("=" * 50)
        print("• Random movement with obstacle avoidance")
        print("• Checks for plastic bottles every 10 seconds")
        print("• Stops when bottle found")
        print("• Press 'x' to stop manually")
        print("=" * 50)
        
        # Setup terminal for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            self.running = True
            self.last_check_time = time.time()
            self.last_direction_change = time.time()
            
            while self.running and not self.bottle_found:
                current_time = time.time()
                
                # Check for stop key
                if self.check_for_stop_key():
                    print("\n🛑 Stop key pressed - shutting down")
                    break
                
                # Check for bottle every 10 seconds
                if current_time - self.last_check_time >= self.check_interval:
                    print(f"\n⏰ Bottle check time! ({self.check_interval}s interval)")
                    
                    # Stop robot for photo
                    self.stop_robot()
                    time.sleep(0.5)  # Brief pause
                    
                    # Capture and check for bottle
                    image = self.capture_image()
                    if image:
                        bottle_detected = self.check_for_bottle(image)
                        if bottle_detected:
                            print("\n" + "=" * 50)
                            print("🎉 DONE. PLASTIC BOTTLE FOUND!")
                            print("=" * 50)
                            self.bottle_found = True
                            break
                        else:
                            print("❌ No plastic bottle detected - continuing search")
                    else:
                        print("✗ Failed to capture image - continuing search")
                    
                    self.last_check_time = current_time
                    print("🚶 Resuming movement...\n")
                
                # Continue random movement
                direction = self.random_movement_with_avoidance()
                self.move_robot(direction, duration=0.4)
                
                # Small delay
                time.sleep(0.2)
                
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
        
        if self.bottle_found:
            print("✅ Mission completed - bottle found!")
        else:
            print("✅ Search ended")
        
        print("✅ Bruno Bottle Searcher cleanup complete")

def main():
    """Main function"""
    try:
        searcher = BottleSearcher()
        searcher.run()
    except Exception as e:
        print(f"Failed to start Bruno Bottle Searcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()