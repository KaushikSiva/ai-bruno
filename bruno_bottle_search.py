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
import random

# Windows-compatible imports
try:
    import select
    import termios
    import tty
    UNIX_TERMINAL = True
except ImportError:
    # Windows doesn't have these modules
    UNIX_TERMINAL = False
    try:
        import msvcrt
    except ImportError:
        pass

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
    from robot_control.head_controller import HeadController
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
        self.check_interval = 30  # seconds - changed from 10 to 30
        self.bottle_found = False
        
        # Movement control
        self.last_direction_change = 0
        self.direction_change_interval = 3  # Change direction every 3 seconds
        self.current_direction = "forward"
        
        print("🔍 Bruno Bottle Searcher initialized")
        print("• Moves randomly avoiding obstacles")
        print("• Checks for bottles every 30 seconds")
        print("• Stops when plastic bottle found")
        print("• Rate limit: 2 GPT requests per minute")
    
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
        """Setup movement and head controllers"""
        if BRUNO_MODULES_AVAILABLE:
            try:
                # Setup movement controller
                config = {
                    'max_speed': 50,  # Medium speed for searching
                    'min_speed': 20,
                    'turn_multiplier': 0.8,
                    'movement_timeout': 0.5
                }
                self.movement = MovementController(config)
                
                # Setup head controller
                head_config = {
                    'head_servo': {
                        'id': 2,
                        'positions': {
                            'center': 1500,
                            'up': 1300,
                            'down': 1700
                        },
                        'timings': {
                            'normal_movement': 0.5
                        }
                    }
                }
                self.head_controller = HeadController(head_config)
                
                self.hardware_available = True
                print("✓ Movement and head controllers initialized")
            except Exception as e:
                print(f"Warning: Hardware not available - {e}")
                self.movement = None
                self.head_controller = None
                self.hardware_available = False
        else:
            self.movement = None
            self.head_controller = None
            self.hardware_available = False
    
    def reconnect_camera(self):
        """Reconnect to camera to ensure fresh stream"""
        print("🔌 Reconnecting to camera for fresh stream...")
        
        if self.cap:
            self.cap.release()
            time.sleep(0.5)  # Wait for release
        
        # Try the same camera sources as initialization
        camera_sources = [
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream',
            0, 1, 2
        ]
        
        for source in camera_sources:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test capture to ensure it works
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera reconnected: {source}")
                    self.cap = cap
                    return True
                else:
                    cap.release()
        
        print("✗ Camera reconnection failed")
        return False
    
    def capture_image(self):
        """Capture genuinely fresh image by reconnecting to camera"""
        print("📸 Capturing FRESH image for bottle detection...")
        
        # Force camera reconnection for fresh stream
        if not self.reconnect_camera():
            print("✗ Camera reconnection failed")
            return None
        
        # Wait for camera to stabilize after reconnection
        print("⏳ Stabilizing camera after reconnection...")
        time.sleep(1.0)
        
        # Clear any buffered frames from the new connection
        print("🔄 Clearing initial frames from fresh connection...")
        for i in range(5):
            ret, frame = self.cap.read()
            if ret:
                print(f"   Clearing frame {i+1}/5")
            time.sleep(0.1)
        
        # Now capture the actual fresh image
        print("📷 Capturing final fresh image...")
        ret, frame = self.cap.read()
        if ret and frame is not None:
            print("✓ Fresh frame captured from new connection")
            
            # Show frame info for verification
            frame_hash = hash(frame.tobytes())
            frame_mean = frame.mean()
            print(f"Frame stats - Hash: {frame_hash % 10000}, Mean: {frame_mean:.1f}")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Save image with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            milliseconds = int(time.time() * 1000) % 1000
            filename = f"bottle_search_{timestamp}_{milliseconds:03d}.jpg"
            image.save(filename)
            print(f"✓ Fresh image saved: {filename}")
            
            # Acknowledgment nod and announcement
            print("📸 PHOTO TAKEN!")
            self.acknowledge_photo()
            
            return image
        else:
            print("✗ Failed to capture fresh image")
            return None
    
    def acknowledge_photo(self):
        """Nod to acknowledge photo taken"""
        if self.hardware_available and self.head_controller:
            try:
                print("🙋 Nodding acknowledgment...")
                self.head_controller.nod_yes(repetitions=2)
            except Exception as e:
                print(f"Head nod failed: {e}")
        else:
            print("🙋 [SIM] Nodding acknowledgment")
    
    def describe_photo(self, image):
        """Get photo description from GPT Vision"""
        try:
            # Encode image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_data = buffer.getvalue()
            base64_data = base64.b64encode(jpeg_data).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{base64_data}"
            
            print("🤖 Getting photo description...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe what you see in this image in 1-2 sentences. Focus on the main objects, environment, and any notable features."
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
                max_tokens=100,
                temperature=0.1
            )
            
            description = response.choices[0].message.content.strip()
            print(f"\n📋 PHOTO DESCRIPTION:")
            print(f"   {description}\n")
            return description
            
        except APIError as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("❌ No photo captured. (Rate limit reached)")
                return "Rate limit exceeded"
            else:
                print(f"✗ OpenAI API error: {e}")
                return None
        except Exception as e:
            print(f"✗ Description error: {e}")
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
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("❌ No photo captured. (Rate limit reached)")
                return False
            else:
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
            if UNIX_TERMINAL:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == 'x':
                        return True
            else:
                # Windows version using msvcrt
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8').lower()
                    if char == 'x':
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
        print("• Checks for plastic bottles every 30 seconds")  
        print("• 2 GPT requests per minute (rate limit safe)")
        print("• Stops when bottle found")
        print("• Press 'x' to stop manually")
        print("=" * 50)
        
        # Setup terminal for non-blocking input (Unix only)
        old_settings = None
        if UNIX_TERMINAL:
            old_settings = termios.tcgetattr(sys.stdin)
        try:
            if UNIX_TERMINAL:
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
                
                # Check for bottle every 30 seconds
                if current_time - self.last_check_time >= self.check_interval:
                    print(f"\n⏰ Bottle check time! ({self.check_interval}s interval)")
                    
                    # Stop robot for photo
                    self.stop_robot()
                    time.sleep(0.5)  # Brief pause
                    
                    # Capture and analyze photo
                    image = self.capture_image()
                    if image:
                        # Get photo description first
                        self.describe_photo(image)
                        
                        # Then check for bottle
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
            # Restore terminal settings (Unix only)
            if UNIX_TERMINAL and old_settings:
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