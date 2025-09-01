#!/usr/bin/env python3
# coding: utf-8
"""
Local HuggingFace Vision Test Script
- Tests image captioning using transformers library locally
- Downloads model once and runs locally (no API needed)
- Usage: python test_hf_vision_local.py <image_path>
"""

import os
import sys
from PIL import Image

print("🤖 Local HuggingFace Vision Test")
print("=" * 50)

# Check if transformers is installed
try:
    from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
    print("✅ Transformers library available")
except ImportError:
    print("❌ Transformers library not installed")
    print("Install with: pip install transformers torch pillow")
    sys.exit(1)

def test_local_vision(image_path: str):
    """Test local HuggingFace vision model"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        return False
    
    print(f"✓ Image file found: {image_path}")
    
    try:
        # Load image
        print("📷 Loading image...")
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]}")
        
        print("\n🧠 Method 1: Using transformers pipeline (automatic)")
        print("-" * 60)
        
        try:
            # Create image-to-text pipeline
            print("⏳ Loading image-to-text pipeline (first time may take a while)...")
            pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            
            print("🔍 Generating caption...")
            result = pipe(image)
            
            if result and len(result) > 0:
                caption = result[0]['generated_text']
                print(f"✅ SUCCESS!")
                print(f"🔍 DESCRIPTION: {caption}")
                return True
            else:
                print(f"⚠️ No caption generated")
                
        except Exception as e:
            print(f"❌ Pipeline method failed: {e}")
        
        print("\n🧠 Method 2: Direct model loading")
        print("-" * 60)
        
        try:
            print("⏳ Loading BLIP model directly...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            print("🔍 Processing image...")
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"✅ SUCCESS!")
            print(f"🔍 DESCRIPTION: {caption}")
            return True
            
        except Exception as e:
            print(f"❌ Direct model method failed: {e}")
        
        print("\n❌ Both methods failed")
        return False
        
    except Exception as e:
        print(f"❌ ERROR processing image: {e}")
        return False

def create_test_image():
    """Create a simple test image if none provided"""
    test_image_path = "test_image_local.jpg"
    
    # Create a simple colored image
    img = Image.new('RGB', (400, 300), color='lightblue')
    
    # Add some basic shapes using PIL drawing
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw shapes
        draw.rectangle([50, 50, 150, 100], fill='red')
        draw.ellipse([200, 80, 350, 180], fill='green')
        draw.rectangle([100, 200, 300, 250], fill='yellow')
        
    except Exception as e:
        print(f"⚠️ Could not add shapes: {e}")
    
    img.save(test_image_path)
    print(f"✓ Created test image: {test_image_path}")
    return test_image_path

def main():
    # Get image path from command line or create test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("No image provided, creating test image...")
        image_path = create_test_image()
    
    # Test the local vision model
    success = test_local_vision(image_path)
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Local HuggingFace Vision Test: SUCCESS!")
        print("✅ Local image captioning is working!")
        print("💡 You can now use this approach in your Bruno scripts")
    else:
        print("❌ Local HuggingFace Vision Test: FAILED!")
        print("💡 Try installing missing dependencies:")
        print("   pip install transformers torch pillow")
        print("   pip install accelerate  # For faster loading")

if __name__ == "__main__":
    main()