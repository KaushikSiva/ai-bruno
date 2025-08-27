#!/usr/bin/env python3
"""
Simple Vision Test Script - Camera Capture Only
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

def setup_camera():
    """Setup camera capture"""
    try:
        # Try multiple camera sources
        camera_sources = [
            0,
            'http://127.0.0.1:8080?action=stream',
            'http://localhost:8080?action=stream',
            1, 2, 3
        ]
        
        for source in camera_sources:
            print(f"Trying camera source: {source}")
            cap = cv2.VideoCapture(source)
            
            if cap.isOpened():
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test capture
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera connected: {source}")
                    return cap
                else:
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
    
    print("Capturing fresh image from camera...")
    
    # Flush camera buffer to get the most recent frame
    print("Flushing camera buffer...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"   Flushed frame {i+1}/5")
        time.sleep(0.05)  # Small delay
    
    # Capture the actual fresh frame
    print("Capturing actual image...")
    ret, frame = cap.read()
    if ret:
        print("✓ Fresh frame captured")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_frame)
        return image
    else:
        print("✗ Failed to capture fresh frame")
        return None

def encode_image_for_gpt(image):
    """Encode PIL image to base64 for GPT Vision"""
    # Convert to JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    jpeg_data = buffer.getvalue()
    
    # Encode to base64
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
        print(f"✗ OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def main():
    """Main function - capture and analyze"""
    print("Simple Vision Test - Camera Capture")
    print("=" * 40)
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Setup camera
    print("Setting up camera...")
    cap = setup_camera()
    if cap is None:
        print("✗ Camera setup failed")
        sys.exit(1)
    
    try:
        # Capture image
        print("Capturing image from camera...")
        image = capture_image_from_camera(cap)
        
        if image is None:
            print("✗ Failed to capture image")
            sys.exit(1)
        
        # Save image
        image.save("captured_image.jpg")
        print("✓ Image saved as captured_image.jpg")
        
        # Encode for GPT Vision
        image_base64 = encode_image_for_gpt(image)
        
        # Ask GPT Vision
        print("Asking GPT Vision...")
        description = ask_gpt_vision(client, image_base64, 
                                   "Describe what you see in this image. Look for bottles, bins, obstacles, and any other objects.")
        
        if description:
            print("\n" + "=" * 60)
            print("GPT VISION DESCRIPTION")
            print("=" * 60)
            print(description)
            print("=" * 60)
        else:
            print("✗ No description received")
            
    finally:
        if cap:
            cap.release()
            print("Camera released")

if __name__ == "__main__":
    main()