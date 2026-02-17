from typing import Any, Dict, List

from bruno_core.config.env import get_env_str
from bruno_core.inference.providers.openai_compat import extract_text, post_chat_completion
from bruno_core.logging.setup import LOG
from bruno_core.runtime.paths import paths

LLM_API_BASE = get_env_str("LLM_API_BASE")
LLM_MODEL = get_env_str("LLM_MODEL")
LLM_TIMEOUT = float(get_env_str("LLM_TIMEOUT_SEC"))
LLM_API_KEY = get_env_str("LLM_API_KEY")

def _extract_content(data: Dict[str, Any]) -> str:
    text = extract_text(data)
    if text:
        return text
    if isinstance(data.get("error"), dict):
        msg = data["error"].get("message") or data["error"].get("error") or str(data["error"])
        return f"[LM Studio error] {msg}"
    if isinstance(data.get("error"), str):
        return f"[LM Studio error] {data['error']}"
    return f"[Unexpected response format] keys={list(data.keys())}"

def summarize_captions(captions: List[Dict]) -> str:
    if not captions:
        return "No captions were captured in the time window."
    bullet_lines = [f"- [{c['timestamp']}] {c['caption']}" for c in captions]
    user_text = (
        "You are a concise surveillance summarizer.\n"
        "Given the following image captions collected over ~2 minutes, "
        "summarize the scene in 3–6 bullet points.\n\n"
        "CAPTIONS:\n" + "\n".join(bullet_lines)
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You write crisp, factual summaries for surveillance operators."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        data = post_chat_completion(
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY,
            payload=payload,
            timeout_sec=LLM_TIMEOUT,
        )
        try:
            (paths.debug / "lmstudio_last.json").write_text(str(data), encoding="utf-8")
        except Exception as e:
            LOG.warning(f"Failed to write debug LM Studio response: {e}")
        return _extract_content(data)
    except Exception as e:
        return f'[LM Studio summary failed] {e}'


def summarize_captions_lmstudio(captions: List[Dict]) -> str:
    return summarize_captions(captions)
