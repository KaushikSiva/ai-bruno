import os
import base64
from typing import Optional, Tuple, Dict, Any
import requests

from bruno_core.config.env import get_env_str
from bruno_core.logging.setup import LOG


# Environment
GROQ_API_KEY = get_env_str('GROQ_API_KEY')
GROQ_API_BASE = get_env_str('GROQ_API_BASE').rstrip('/')
GROQ_VISION_MODEL = get_env_str('GROQ_VISION_MODEL')


def _file_to_data_url(path: str) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        # Heuristic mime type by suffix
        ext = (os.path.splitext(path)[1] or '').lower()
        mime = 'image/jpeg'
        if ext in ('.png',):
            mime = 'image/png'
        elif ext in ('.webp',):
            mime = 'image/webp'
        return f'data:{mime};base64,{b64}'
    except Exception as e:
        LOG.warning(f'groq_vision: failed to read image {path}: {e}')
        return None


def get_caption(image_path: str, prompt: Optional[str] = None) -> str:
    """
    Generate a caption via Groq's OpenAI-compatible chat API using a vision model.
    Default model can be overridden via GROQ_VISION_MODEL.

    Returns a human-readable string even on error; never raises.
    """
    if not GROQ_API_KEY:
        return 'Groq API key not set (GROQ_API_KEY)'

    data_url = _file_to_data_url(image_path)
    if not data_url:
        return f'Unable to read image: {image_path}'

    user_prompt = prompt or (
        'Describe this image succinctly. Focus on: objects, people, pets, '
        'furniture, containers/bottles, obstacles, walls, and overall scene.'
    )

    payload = {
        'model': GROQ_VISION_MODEL,
        'messages': [
            {'role': 'system', 'content': 'You are a precise visual describer.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': user_prompt},
                {'type': 'image_url', 'image_url': {'url': data_url}},
            ]}
        ],
        'temperature': 0.2,
        'max_tokens': 256,
    }

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }

    try:
        url = f'{GROQ_API_BASE}/chat/completions'
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data['choices'][0]['message']['content'] or '').strip() or 'No description produced.'
    except Exception as e:
        return f'[groq vision caption failed] {e}'


def detect_person_legs(image_path: str) -> Optional[Dict[str, int]]:
    """
    Ask Groq Vision to return a tight bounding box around a person's lower body
    (hips through feet). Returns a pixel-space bbox dict {x,y,w,h} or None.
    """
    if not GROQ_API_KEY:
        LOG.warning('Groq API key not set (GROQ_API_KEY)')
        return None


def detect_object_bbox(image_path: str, target: str = "bottle or can") -> Optional[Dict[str, int]]:
    """
    Ask Groq Vision (Meta Llama-4 Maverick) to return a tight bounding box
    around the most prominent target object. Returns {x,y,w,h} or None.

    - target: freeform string, e.g. "bottle", "aluminum can", or "bottle or can".
    """
    if not GROQ_API_KEY:
        LOG.warning('Groq API key not set (GROQ_API_KEY)')
        return None

    data_url = _file_to_data_url(image_path)
    if not data_url:
        return None

    system = (
        'You are a vision detector. Reply ONLY with minified JSON. '
        f'Detect the TIGHTEST bounding box around the most prominent {target}. '
        'If not present, return {"bbox": null}. '
        'JSON schema: {"bbox": {"x": int, "y": int, "w": int, "h": int}}.'
    )
    user_parts = [
        {"type": "text", "text": f"Return only JSON for bbox of: {target}."},
        {"type": "input_image", "image_url": {"url": data_url}},
    ]
    payload = {
        'model': GROQ_VISION_MODEL,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_parts},
        ],
        'temperature': 0.0,
        'max_tokens': 200,
    }
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    try:
        url = f'{GROQ_API_BASE}/chat/completions'
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        txt = (data['choices'][0]['message']['content'] or '').strip()
        # Strip code fences if present
        if txt.startswith('```'):
            idx = txt.find('\n')
            if idx >= 0:
                txt = txt[idx+1:]
            if txt.endswith('```'):
                txt = txt[:-3]
            txt = txt.strip()
        import json as _json
        obj = _json.loads(txt)
        bbox = obj.get('bbox') if isinstance(obj, dict) else None
        if not bbox:
            return None
        x = int(bbox.get('x', 0)); y = int(bbox.get('y', 0))
        w = int(bbox.get('w', 0)); h = int(bbox.get('h', 0))
        if w <= 0 or h <= 0:
            return None
        return {'x': x, 'y': y, 'w': w, 'h': h}
    except Exception as e:
        LOG.warning(f'groq object detect failed: {e}')
        return None

    # Use data URL for inline image
    data_url = _file_to_data_url(image_path)
    if not data_url:
        return None

    system = (
        'You are a vision detector. Reply ONLY with minified JSON. '
        'Detect the TIGHTEST bounding box around the lower body of the most prominent person '
        '(hips through feet/ankles). If no person found, return {"bbox": null}. '
        'JSON schema: {"bbox": {"x": int, "y": int, "w": int, "h": int}}.'
    )
    user_parts = [
        {"type": "text", "text": "Return only JSON for lower-body bbox as specified."},
        {"type": "input_image", "image_url": {"url": data_url}},
    ]
    payload = {
        'model': GROQ_VISION_MODEL,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_parts},
        ],
        'temperature': 0.0,
        'max_tokens': 200,
    }
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    try:
        url = f'{GROQ_API_BASE}/chat/completions'
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        txt = (data['choices'][0]['message']['content'] or '').strip()
        # Strip possible code fences
        if txt.startswith('```'):
            idx = txt.find('\n')
            if idx >= 0:
                txt = txt[idx+1:]
            if txt.endswith('```'):
                txt = txt[:-3]
            txt = txt.strip()
        import json as _json
        obj = _json.loads(txt)
        bbox = obj.get('bbox') if isinstance(obj, dict) else None
        if not bbox:
            return None
        try:
            x = int(bbox.get('x', 0)); y = int(bbox.get('y', 0))
            w = int(bbox.get('w', 0)); h = int(bbox.get('h', 0))
            if w <= 0 or h <= 0:
                return None
            return {'x': x, 'y': y, 'w': w, 'h': h}
        except Exception:
            return None
    except Exception as e:
        LOG.warning(f'groq legs detect failed: {e}')
        return None
