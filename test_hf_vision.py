#!/usr/bin/env python3
# coding: utf-8
"""
Simple Hugging Face Vision Test Script
- Tests HF_TOKEN and vision models
- Given an image, describes it using HuggingFace BLIP model
- Usage: python test_hf_vision.py <image_path>
"""

import os
import sys
import time
import requests
import base64
from PIL import Image
import io

# Load environment variables from .env file
if os.path.exists('.env'):
    print("📁 Found .env file, loading...")
    with open('.env', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    os.environ[key] = value
                    print(f"  Line {i}: Set {key}={value[:10]}...")
                except ValueError:
                    print(f"  Line {i}: Skipped invalid line: {line}")
    print("✓ Loaded .env file")
    
    # Debug: Show all environment variables that start with HF
    print("\n🔍 Debug: Environment variables starting with HF:")
    for key, value in os.environ.items():
        if key.startswith('HF'):
            print(f"  {key}={value[:10]}...")
else:
    print("❌ .env file not found")

def test_hf_vision(image_path: str):
    """Test Hugging Face vision model with a given image"""
    
    # Check HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ ERROR: HF_TOKEN not found in environment variables")
        print("Please add HF_TOKEN=your_token_here to your .env file")
        print("Get token from: https://huggingface.co/settings/tokens")
        return False
    
    print(f"✓ HF_TOKEN found (length: {len(hf_token)})")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        return False
    
    print(f"✓ Image file found: {image_path}")
    
    try:
        # Load and process image
        print("📷 Loading image...")
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]}")
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        print(f"✓ Image converted to JPEG ({len(img_bytes)} bytes)")
        
        # Test different HuggingFace models - try various available ones
        models_to_test = [
            # Image-to-text models that might be available
            "Salesforce/blip2-opt-2.7b",
            "microsoft/git-base",
            "microsoft/DialoGPT-large", 
            "facebook/detr-resnet-50",
            "google/vit-base-patch16-224",
            # Fallback to any model that accepts images
            "openai/clip-vit-base-patch32",
            "google/owlvit-base-patch32",
        ]
        
        for model in models_to_test:
            print(f"\n🧠 Testing model: {model}")
            print("-" * 60)
            
            # Prepare API request
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "x-wait-for-model": "true",  # Wait for model to load if needed
            }
            
            try:
                # Send request
                print(f"📡 Sending request to: {api_url}")
                start_time = time.time()
                
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    data=img_bytes, 
                    timeout=60
                )
                
                end_time = time.time()
                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                
                # Check response
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ SUCCESS! Status: {response.status_code}")
                    print(f"Raw response: {str(result)[:200]}...")
                    
                    # Parse description
                    description = None
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'generated_text' in result[0]:
                            description = result[0]['generated_text']
                        elif isinstance(result[0], str):
                            description = result[0]
                    elif isinstance(result, dict):
                        description = result.get('generated_text', '') or result.get('text', '') or result.get('caption', '') or str(result)
                    
                    if description and description.strip():
                        print(f"🔍 DESCRIPTION: {description.strip()}")
                        print(f"✅ Model {model} is working perfectly!")
                        return True
                    else:
                        print(f"⚠️  No description found in response: {result}")
                
                elif response.status_code == 404:
                    print(f"❌ Model not found (404) - {model} may not be available via Inference API")
                elif response.status_code == 503:
                    print(f"⏳ Model loading (503) - {model} is starting up, try again in a few minutes")
                elif response.status_code == 429:
                    print(f"🛑 Rate limited (429) - too many requests")
                else:
                    print(f"❌ FAILED! Status: {response.status_code}")
                    print(f"Response: {response.text[:300]}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ TIMEOUT - {model} took too long to respond")
            except Exception as e:
                print(f"❌ ERROR with {model}: {e}")
                
        print(f"\n❌ All models failed or unavailable.")
        print("💡 Try these solutions:")
        print("1. Wait a few minutes for models to load")
        print("2. Check if your HF_TOKEN has inference permissions")
        print("3. Try a different model manually")
        return False
        
    except Exception as e:
        print(f"❌ ERROR processing image: {e}")
        return False

def create_test_image():
    """Create a simple test image if none provided"""
    test_image_path = "test_image.jpg"
    
    # Create a simple colored image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a 640x480 image with gradient background
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 250, 150], fill='red', outline='black', width=3)
    draw.ellipse([350, 100, 550, 300], fill='green', outline='black', width=3)
    draw.rectangle([200, 300, 400, 400], fill='yellow', outline='black', width=3)
    
    # Add text
    try:
        # Try to use default font
        draw.text((100, 200), "TEST IMAGE", fill='black', anchor="mm")
        draw.text((100, 250), "For HuggingFace", fill='black', anchor="mm")
        draw.text((100, 300), "Vision Model", fill='black', anchor="mm")
    except:
        # If font fails, just use basic text
        draw.text((100, 250), "TEST IMAGE", fill='black')
    
    img.save(test_image_path)
    print(f"✓ Created test image: {test_image_path}")
    return test_image_path

def main():
    print("🤖 HuggingFace Vision Model Test")
    print("=" * 50)
    
    # Get image path from command line or create test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("No image provided, creating test image...")
        image_path = create_test_image()
    
    # Test the vision model
    success = test_hf_vision(image_path)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 HuggingFace Vision Test: SUCCESS!")
        print("✅ Your HF_TOKEN and vision models are working correctly.")
    else:
        print("❌ HuggingFace Vision Test: FAILED!")
        print("Please check:")
        print("1. HF_TOKEN is correct in your .env file")
        print("2. Token has read permissions")
        print("3. Internet connection is working")
        print("4. Try getting a new token from: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()