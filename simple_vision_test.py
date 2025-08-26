#!/usr/bin/env python3
"""
Simple Vision Test Script - OpenAI Only
Tests OpenAI Vision API with existing images or URLs.
No camera hardware required.
"""

import os
import sys
import time
import json
import argparse
import base64
import io
import requests
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Error: OpenAI library not available. Install with: pip install openai")
    sys.exit(1)

def load_image_from_file(image_path):
    """Load image from local file"""
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        # Load image with PIL
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"✓ Image loaded: {image_path} ({image.size[0]}x{image.size[1]} pixels)")
        return image
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_image_from_url(image_url):
    """Load image from URL"""
    try:
        print(f"Downloading image from: {image_url}")
        
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Load image from bytes
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"✓ Image downloaded: {image_url} ({image.size[0]}x{image.size[1]} pixels)")
        return image
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def encode_image_for_gpt(image, max_width=800, quality=85):
    """Encode image for GPT Vision API"""
    # Resize if too large
    width, height = image.size
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
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

def setup_camera(device_id=0):
    """Setup camera capture - similar to gpt.py approach"""
    try:
        import cv2
        
        # Try multiple camera sources like gpt.py
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

def capture_image_from_camera(cap, save_path=None):
    """Capture an image from the camera"""
    if cap is None:
        print("No camera available")
        return None
    
    print("Capturing image from camera...")
    
    # Capture multiple frames to ensure we get a good one
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured (attempt {i+1})")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_frame)
            
            # Save image if requested
            if save_path:
                image.save(save_path)
                print(f"Image saved to: {save_path}")
            
            return image
        else:
            print(f"✗ Frame capture failed (attempt {i+1})")
            time.sleep(0.1)
    
    print("✗ Failed to capture image after 5 attempts")
    return None

def create_test_image():
    """Create a simple test image for testing"""
    try:
        # Create a simple test image with some shapes
        width, height = 640, 480
        image = Image.new('RGB', (width, height), color='lightblue')
        
        # Draw some simple shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Draw a red circle (bottle-like)
        draw.ellipse([200, 150, 300, 350], fill='red', outline='darkred', width=3)
        
        # Draw a green rectangle (bin-like)
        draw.rectangle([400, 200, 550, 400], fill='green', outline='darkgreen', width=3)
        
        # Add some text
        draw.text((50, 50), "Test Image", fill='black')
        draw.text((50, 80), "Red circle = bottle", fill='red')
        draw.text((50, 110), "Green rectangle = bin", fill='green')
        
        print("✓ Test image created")
        return image
        
    except Exception as e:
        print(f"Error creating test image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Simple Vision Test - OpenAI Vision API Only")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--url", help="URL to image")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--capture", action="store_true", help="Capture image from camera")
    parser.add_argument("--create-test", action="store_true", help="Create a test image")
    parser.add_argument("--save", help="Save processed image to file")
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
    print("SIMPLE VISION TEST - OPENAI VISION API")
    print("=" * 60)
    if args.capture:
        print(f"Image source: Camera device {args.camera}")
    elif args.image:
        print(f"Image source: File {args.image}")
    elif args.url:
        print(f"Image source: URL {args.url}")
    elif args.create_test:
        print("Image source: Generated test image")
    else:
        print("Image source: Generated test image (default)")
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
    
        # Load or create image
    image = None
    cap = None
    
    if args.capture:
        print("Setting up camera...")
        cap = setup_camera(args.camera)
        if cap is None:
            print("✗ Camera setup failed")
            sys.exit(1)
        
        print("Capturing image from camera...")
        image = capture_image_from_camera(cap, args.save)
        
    elif args.create_test:
        print("Creating test image...")
        image = create_test_image()
        
    elif args.image:
        print(f"Loading image from file: {args.image}")
        image = load_image_from_file(args.image)
        
    elif args.url:
        print(f"Loading image from URL: {args.url}")
        image = load_image_from_url(args.url)
        
    else:
        print("No image source specified. Creating test image...")
        image = create_test_image()
    
    if image is None:
        print("✗ Failed to load/create image")
        sys.exit(1)

    try:
        # Save image if requested
        if args.save and not args.capture:  # Don't save twice if already saved during capture
            image.save(args.save)
            print(f"✓ Image saved to: {args.save}")
        
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
