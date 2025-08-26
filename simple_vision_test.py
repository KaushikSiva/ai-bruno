#!/usr/bin/env python3
"""
Simple Vision Test Script
Takes a picture and asks OpenAI Vision to describe what it sees.
Useful for testing camera and GPT Vision integration.
"""

import os
import sys
import time
import json
import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Error: OpenAI library not available. Install with: pip install openai")
    sys.exit(1)

def setup_camera(device_id=0):
    """Setup camera capture"""
    try:
        import cv2
        
        # Try multiple camera sources
        camera_sources = [
            device_id,
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
                
                # Test frame capture
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera connected: {source}")
                    return cap
                else:
                    cap.release()
            else:
                cap.release()
        
        print("✗ No camera sources available")
        return None
        
    except ImportError:
        print("Error: OpenCV not available. Install with: pip install opencv-python")
        return None
    except Exception as e:
        print(f"Error setting up camera: {e}")
        return None

def capture_image(cap, save_path=None):
    """Capture an image from the camera"""
    if cap is None:
        print("No camera available")
        return None
    
    print("Capturing image...")
    
    # Capture multiple frames to ensure we get a good one
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured (attempt {i+1})")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save image if requested
            if save_path:
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(save_path)
                print(f"Image saved to: {save_path}")
            
            return rgb_frame
        else:
            print(f"✗ Frame capture failed (attempt {i+1})")
            time.sleep(0.1)
    
    print("✗ Failed to capture image after 5 attempts")
    return None

def encode_image_for_gpt(image_array, max_width=800, quality=85):
    """Encode image for GPT Vision API"""
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Resize if too large
    width, height = pil_image.size
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to JPEG bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = buffer.getvalue()
    
    # Encode to base64
    base64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
    
    print(f"Image encoded: {len(jpeg_bytes)} bytes, {len(base64_image)} base64 chars")
    return base64_image

def ask_gpt_vision(client, image_base64, prompt="Describe what you see in this image in detail."):
    """Ask GPT Vision to describe the image"""
    try:
        print("Sending image to GPT Vision...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        description = response.choices[0].message.content
        print("✓ GPT Vision response received")
        return description
        
    except APIError as e:
        print(f"✗ OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Simple Vision Test - Take picture and get GPT Vision description")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--save", help="Save captured image to file")
    parser.add_argument("--prompt", default="Describe what you see in this image in detail. Focus on any objects, people, or interesting features.",
                       help="Custom prompt for GPT Vision")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    print("=" * 60)
    print("SIMPLE VISION TEST")
    print("=" * 60)
    print(f"Camera device: {args.camera}")
    print(f"Save image: {args.save or 'No'}")
    print(f"Prompt: {args.prompt[:50]}...")
    print("=" * 60)
    
    # Setup OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    
    # Setup camera
    cap = setup_camera(args.camera)
    if cap is None:
        print("✗ Camera setup failed")
        sys.exit(1)
    
    try:
        # Capture image
        image = capture_image(cap, args.save)
        if image is None:
            print("✗ Image capture failed")
            sys.exit(1)
        
        print(f"✓ Image captured: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Encode image for GPT Vision
        image_base64 = encode_image_for_gpt(image)
        
        # Ask GPT Vision for description
        description = ask_gpt_vision(client, image_base64, args.prompt)
        
        if description:
            print("\n" + "=" * 60)
            print("GPT VISION DESCRIPTION")
            print("=" * 60)
            print(description)
            print("=" * 60)
        else:
            print("✗ Failed to get description from GPT Vision")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap:
            cap.release()
            print("Camera released")

if __name__ == "__main__":
    main()
