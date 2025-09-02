import os, requests
from typing import List, Dict, Any
from utils import get_env_str, paths, LOG

LLM_API_BASE = get_env_str('LLM_API_BASE', 'http://localhost:1234/v1')
LLM_MODEL    = get_env_str('LLM_MODEL', 'lmstudio')
LLM_TIMEOUT  = float(get_env_str('LLM_TIMEOUT_SEC', '30'))
LLM_API_KEY  = get_env_str('LLM_API_KEY', 'lm-studio')
LLM_ENDPOINT = get_env_str('LLM_ENDPOINT', 'chat').strip().lower()  # 'chat' or 'completions'

def _extract_content(data: Dict[str, Any]) -> str:
    try: return data['choices'][0]['message']['content'].strip()
    except Exception: pass
    try: return data['choices'][0]['text'].strip()
    except Exception: pass
    if isinstance(data.get('text'), str): return data['text'].strip()
    if isinstance(data.get('error'), dict):
        msg = data['error'].get('message') or data['error'].get('error') or str(data['error'])
        return f'[LM Studio error] {msg}'
    if isinstance(data.get('error'), str): return f'[LM Studio error] {data["error"]}'
    return f'[Unexpected response format] keys={list(data.keys())}'

def summarize_captions_lmstudio(captions: List[Dict]) -> str:
    if not captions: return 'No captions were captured in the time window.'
    bullet_lines = [f"- [{c['timestamp']}] {c['caption']}" for c in captions]
    user_text = ('You are a concise surveillance summarizer.\n'
                 'Given the following image captions collected over ~2 minutes, '
                 'summarize the scene in 3–6 bullet points.\n\n'
                 'CAPTIONS:\n' + "\n".join(bullet_lines))
    payload = ({'model': LLM_MODEL, 'messages': [
                {'role':'system','content':'You write crisp, factual summaries for surveillance operators.'},
                {'role':'user','content': user_text}], 'temperature':0.2, 'max_tokens':400}
               if LLM_ENDPOINT=='chat' else
               {'model': LLM_MODEL, 'prompt': user_text, 'temperature':0.2, 'max_tokens':400})
    path = '/chat/completions' if LLM_ENDPOINT=='chat' else '/completions'
    url = LLM_API_BASE.rstrip('/') + path
    headers = {'Content-Type':'application/json','Authorization': f'Bearer {LLM_API_KEY}'}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        try: (paths.debug/'lmstudio_last.json').write_text(r.text, encoding='utf-8')
        except Exception as e: LOG.warning(f'Failed to write debug LM Studio response: {e}')
        r.raise_for_status()
        return _extract_content(r.json())
    except Exception as e:
        return f'[LM Studio summary failed] {e}'
