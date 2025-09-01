#!/usr/bin/env python3
# coding: utf-8
"""
HuggingFace BLIP Vision Test Script
- Based on official HuggingFace BLIP documentation
- Uses BlipForConditionalGeneration for image captioning
- Usage: python test_hf_vision.py <image_path>
"""

import os
import sys
import time
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

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
else:
    print("⚠️ .env file not found")

def test_blip_vision(image_path: str):
    """Test BLIP vision model using official HuggingFace documentation approach"""
    
    print("🤖 HuggingFace BLIP Vision Test")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        return False
    
    print(f"✓ Image file found: {image_path}")
    
    try:
        # Load image (following official docs)
        print("📷 Loading image...")
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]}")
        
        # Load the model and processor (following official docs)
        print("⏳ Loading BLIP model and processor...")
        print("   (First time will download ~500MB model)")
        start_time = time.time()
        
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test 1: Unconditional image captioning (as per docs)
        print("\n🔍 Test 1: Unconditional Image Captioning")
        print("-" * 40)
        
        print("Processing image...")
        inputs = processor(images=image, return_tensors="pt")
        
        print("Generating caption...")
        start_time = time.time()
        outputs = model.generate(**inputs, max_length=50, num_beams=5)
        generation_time = time.time() - start_time
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"⏱️ Generation time: {generation_time:.2f} seconds")
        print(f"🔍 UNCONDITIONAL CAPTION: {caption}")
        
        # Test 2: Conditional image captioning with prompt (as per docs)
        print("\n🔍 Test 2: Conditional Image Captioning")
        print("-" * 40)
        
        text_prompts = [
            "A picture of",
            "This image shows",
            "In this photo"
        ]
        
        for prompt in text_prompts:
            print(f"📝 Using prompt: '{prompt}'")
            
            # Process the image and text (following official docs)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            
            # Generate caption
            start_time = time.time()
            outputs = model.generate(**inputs, max_length=50, num_beams=5)
            generation_time = time.time() - start_time
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   ⏱️ Time: {generation_time:.2f}s")
            print(f"   🔍 CAPTION: {caption}")
            print()
        
        # Test 3: Multiple generation parameters
        print("🔍 Test 3: Different Generation Parameters")
        print("-" * 40)
        
        generation_configs = [
            {"max_length": 30, "num_beams": 3, "name": "Short & Fast"},
            {"max_length": 100, "num_beams": 5, "name": "Long & Detailed"},
            {"max_length": 50, "do_sample": True, "temperature": 0.7, "name": "Creative"}
        ]
        
        for config in generation_configs:
            config_name = config.pop("name")
            print(f"🎛️ {config_name}: {config}")
            
            inputs = processor(images=image, return_tensors="pt")
            
            start_time = time.time()
            outputs = model.generate(**inputs, **config)
            generation_time = time.time() - start_time
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   ⏱️ Time: {generation_time:.2f}s")
            print(f"   🔍 CAPTION: {caption}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\n💡 Try installing dependencies:")
        print("   pip install transformers torch pillow requests")
        return False

def create_test_image():
    """Create a simple test image if none provided"""
    test_image_path = "blip_test_image.jpg"
    
    print("🎨 Creating test image...")
    
    # Create a detailed test image
    img = Image.new('RGB', (640, 480), color='skyblue')
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Draw a house scene
        # House
        draw.rectangle([200, 250, 440, 400], fill='brown', outline='black', width=2)
        
        # Roof
        draw.polygon([180, 250, 320, 150, 460, 250], fill='red', outline='black')
        
        # Door
        draw.rectangle([280, 320, 330, 400], fill='darkbrown', outline='black', width=2)
        
        # Windows
        draw.rectangle([220, 280, 260, 320], fill='lightblue', outline='black', width=2)
        draw.rectangle([380, 280, 420, 320], fill='lightblue', outline='black', width=2)
        
        # Sun
        draw.ellipse([500, 50, 580, 130], fill='yellow', outline='orange', width=3)
        
        # Ground
        draw.rectangle([0, 400, 640, 480], fill='green')
        
        # Tree
        draw.rectangle([100, 300, 120, 400], fill='brown')  # Trunk
        draw.ellipse([80, 250, 140, 320], fill='darkgreen')  # Leaves
        
        # Add text
        try:
            draw.text((250, 450), "Test House Scene", fill='black', anchor="mm")
        except:
            draw.text((200, 450), "Test House Scene", fill='black')
            
    except Exception as e:
        print(f"⚠️ Could not add detailed drawing: {e}")
    
    img.save(test_image_path)
    print(f"✅ Created detailed test image: {test_image_path}")
    return test_image_path

def main():
    print("🤖 BLIP Vision Model Test")
    print("Based on official HuggingFace documentation")
    print("=" * 60)
    
    # Get image path from command line or create test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"📁 Using provided image: {image_path}")
    else:
        print("📁 No image provided, creating test image...")
        image_path = create_test_image()
    
    # Test the BLIP vision model
    success = test_blip_vision(image_path)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 BLIP Vision Test: SUCCESS!")
        print("✅ HuggingFace BLIP model is working perfectly!")
        print("💡 Ready for integration into Bruno surveillance system")
        
        print("\n📋 Summary:")
        print("• Model: Salesforce/blip-image-captioning-base")
        print("• Method: BlipForConditionalGeneration")
        print("• Supports: Conditional and unconditional captioning")
        print("• Ready for: Local deployment in Bruno scripts")
        
    else:
        print("❌ BLIP Vision Test: FAILED!")
        print("💡 Install dependencies and try again:")
        print("   pip install transformers torch pillow requests")
        print("   pip install accelerate  # Optional: for faster model loading")

if __name__ == "__main__":
    main()