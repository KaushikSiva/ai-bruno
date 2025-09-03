import os
import base64
from typing import Optional
import requests

from utils import LOG


# Environment
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_BASE = os.environ.get('GROQ_API_BASE', 'https://api.groq.com/openai/v1').rstrip('/')
GROQ_VISION_MODEL = os.environ.get('GROQ_VISION_MODEL', 'llama-4-maverick-17b-128e-instruct')


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
                {'type': 'input_image', 'image_url': {'url': data_url}},
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

