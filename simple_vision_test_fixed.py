#!/usr/bin/env python3
"""
Simple Vision Test Script - Fixed for Windows External Camera
Captures image from camera and tests OpenAI Vision API.
"""

import os
import sys
import time
import base64
import io

from PIL import Image

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

def detect_cameras():
    """Detect available cameras on Windows"""
    print("Detecting available cameras...")
    working_cameras = []
    
    # Test with DirectShow backend (Windows specific)
    for device_id in range(5):
        try:
            print(f"Testing camera {device_id}...", end=" ")
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"[OK] {width}x{height}")
                    working_cameras.append({
                        'device_id': device_id,
                        'width': width,
                        'height': height
                    })
                    cap.release()
                else:
                    print("[FAIL] No frame")
                    cap.release()
            else:
                print("[FAIL] Cannot open")
                
        except Exception as e:
            print(f"[ERROR] {e}")
    
    return working_cameras

def setup_camera():
    """Setup camera capture with Windows DirectShow"""
    try:
        # First detect available cameras
        cameras = detect_cameras()
        
        if not cameras:
            print("No cameras detected!")
            return None
        
        print(f"\nFound {len(cameras)} cameras:")
        for cam in cameras:
            print(f"  Device {cam['device_id']}: {cam['width']}x{cam['height']}")
        
        # Use the first available camera (or ask user to choose)
        selected_camera = cameras[0]
        if len(cameras) > 1:
            print(f"\nUsing camera {selected_camera['device_id']} (first available)")
            print("To use a different camera, modify device_id in the script")
        
        # Open the selected camera with DirectShow
        cap = cv2.VideoCapture(selected_camera['device_id'], cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"Failed to open camera {selected_camera['device_id']}")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test capture
        ret, frame = cap.read()
        if ret:
            print(f"[OK] Camera {selected_camera['device_id']} ready for capture")
            return cap
        else:
            print(f"[FAIL] Cannot capture from camera {selected_camera['device_id']}")
            cap.release()
            return None
            
    except Exception as e:
        print(f"Error setting up camera: {e}")
        return None

def capture_image_from_camera(cap):
    """Capture a fresh image from the camera"""
    if cap is None:
        print("No camera available")
        return None
    
    print("Capturing image from camera...")
    
    # For USB cameras, just capture a few frames to ensure freshness
    for i in range(3):
        ret, frame = cap.read()
        if ret and frame is not None:
            time.sleep(0.1)  # Short delay
        else:
            print(f"Failed to read frame {i+1}")
    
    # Capture the final frame
    ret, frame = cap.read()
    if ret and frame is not None:
        print("[OK] Image captured successfully")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_frame)
        return image
    else:
        print("[FAIL] Failed to capture image")
        return None

def show_camera_preview(cap, duration=5):
    """Show a live preview of the camera for testing"""
    if cap is None:
        print("No camera available for preview")
        return
    
    print(f"Showing camera preview for {duration} seconds...")
    print("Press 'q' to quit early or wait for timeout")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"Camera Preview - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Preview', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame")
            break
    
    cv2.destroyAllWindows()
    print("Preview ended")

def encode_image_for_gpt(image):
    """Encode PIL image to base64 for GPT Vision"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    jpeg_data = buffer.getvalue()
    
    base64_data = base64.b64encode(jpeg_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def ask_gpt_vision(client, image_base64, prompt="What do you see in this image?"):
    """Ask GPT Vision to describe the image"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64,
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0
        )
        
        return response.choices[0].message.content
        
    except APIError as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None

def main():
    """Main function - capture and analyze"""
    print("Simple Vision Test - External Camera Fixed")
    print("=" * 50)
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Setup camera
    cap = setup_camera()
    if cap is None:
        print("[ERROR] Camera setup failed")
        print("\nTroubleshooting:")
        print("1. Make sure your external camera is connected")
        print("2. Check if another application is using the camera")
        print("3. Try a different USB port")
        sys.exit(1)
    
    try:
        # Show preview first
        preview = input("\nShow camera preview? (y/n): ").lower()
        if preview == 'y':
            show_camera_preview(cap, duration=5)
        
        # Capture image
        print("\nCapturing image from camera...")
        image = capture_image_from_camera(cap)
        
        if image is None:
            print("[ERROR] Failed to capture image")
            sys.exit(1)
        
        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_image_{timestamp}.jpg"
        image.save(filename)
        print(f"[OK] Image saved as {filename}")
        
        # Encode for GPT Vision
        image_base64 = encode_image_for_gpt(image)
        
        # Ask GPT Vision
        print("\nAsking GPT Vision...")
        description = ask_gpt_vision(client, image_base64, 
                                   "Describe what you see in this image. Look for bottles, bins, obstacles, and any other objects.")
        
        if description:
            print("\n" + "=" * 60)
            print("GPT VISION DESCRIPTION")
            print("=" * 60)
            print(description)
            print("=" * 60)
        else:
            print("[ERROR] No description received")
            
    finally:
        if cap:
            cap.release()
            print("\nCamera released")

if __name__ == "__main__":
    main()