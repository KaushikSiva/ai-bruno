from typing import Any, Dict

import requests


def post_chat_completion(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


def extract_text(data: Dict[str, Any]) -> str:
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass
    try:
        return (data["choices"][0]["text"] or "").strip()
    except Exception:
        pass
    if isinstance(data.get("text"), str):
        return data["text"].strip()
    return ""

