# Minimal local captioner stub.
# Tries to use Hugging Face transformers pipeline if available.
# You can replace this file with your own captioner.py that exposes get_caption(image_path) -> str

import os

def get_caption(image_path: str) -> str:
    try:
        from transformers import pipeline
        model_name = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
        cap = pipeline("image-to-text", model=model_name)
        out = cap(image_path)
        if isinstance(out, list) and out:
            txt = out[0].get("generated_text") or out[0].get("caption") or str(out[0])
            return txt.strip()
        return "[no caption produced]"
    except Exception as e:
        return f"[captioner unavailable: {e}]"
