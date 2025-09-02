import os, requests
from typing import List, Dict
from utils import get_env_str

LLM_API_BASE = get_env_str('LLM_API_BASE', 'http://localhost:1234/v1')
LLM_MODEL    = get_env_str('LLM_MODEL', 'lmstudio')
LLM_TIMEOUT  = float(get_env_str('LLM_TIMEOUT_SEC', '30'))
LLM_API_KEY  = get_env_str('LLM_API_KEY', 'lm-studio')

def summarize_captions_lmstudio(captions: List[Dict]) -> str:
    if not captions:
        return 'No captions were captured in the time window.'

    bullet_lines = [f"- [{c['timestamp']}] {c['caption']}" for c in captions]
    user_text = (
        'You are a concise surveillance summarizer.\n'
        'Given the following image captions collected over ~2 minutes, '
        'summarize the scene: key objects, activities, potential hazards, and overall context in 3–6 bullet points.\n\n'
        'CAPTIONS:\n' + "\n".join(bullet_lines)
    )

    payload = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': 'You write crisp, factual summaries for surveillance operators.'},
            {'role': 'user',   'content': user_text}
        ],
        'temperature': 0.2,
        'max_tokens': 400
    }

    url = LLM_API_BASE.rstrip('/') + '/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LLM_API_KEY}'
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'[LM Studio summary failed] {e}'
