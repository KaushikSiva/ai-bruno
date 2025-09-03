# File: captioner.py
import os
from typing import Optional
from transformers import pipeline

# You can override the model via env (defaults to BLIP large).
# Tip for low-power devices: try "Salesforce/blip-image-captioning-base"
MODEL_ID = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")

image_to_text_pipeline: Optional[object] = None
try:
    # If you have PyTorch with MPS/CUDA, you can set device=0; we’ll stay device-agnostic.
    image_to_text_pipeline = pipeline("image-to-text", model=MODEL_ID)
    print(f"[captioner] Loaded image-to-text model: {MODEL_ID}")
except Exception as e:
    print(f"[captioner] Error initializing pipeline ({MODEL_ID}): {e}")
    image_to_text_pipeline = None

def get_caption(image_path: str) -> str:
    """
    Generate a caption for an image (local path or URL).
    Returns a human-readable string even on error, never raises.
    """
    if image_to_text_pipeline is None:
        return "Captioning service is unavailable (pipeline not initialized)."

    # If local path, ensure it exists
    if not image_path.startswith(("http://", "https://")) and not os.path.exists(image_path):
        return f"Error: image not found: {image_path}"

    try:
        result = image_to_text_pipeline(image_path)
        # Expected shape: [{'generated_text': '...'}]
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return "No caption generated."
    except Exception as e:
        return f"An error occurred during captioning: {e}"

if __name__ == "__main__":
    # Quick self-test (uses a public image)
    url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    print("[captioner] Testing URL:", url)
    print("[captioner] Caption:", get_caption(url))
