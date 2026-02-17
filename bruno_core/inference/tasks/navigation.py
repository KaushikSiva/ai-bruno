import base64
import json
from typing import Any, Dict, Optional, Tuple

import cv2

ALLOWED_ACTIONS = {"left", "right", "forward", "stop"}


def build_nav_payload(frame, scene_hint: str, model: str) -> Dict[str, Any]:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    prompt = (
        "You control a small rover from a single camera frame. "
        "Choose one safe short action: left, right, forward, or stop. "
        "Return strict minified JSON only with schema "
        '{"action":"left|right|forward|stop","reason":"short","confidence":0.0}. '
        "Use higher confidence only when you are sure. "
        f"Perception hint: {scene_hint}."
    )
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": "Return only minified JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }


def parse_nav_response(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    text = (raw or "").strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None, "json_parse_failed"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    action = str(obj.get("action", "")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        return None, f"invalid_action:{action}"
    reason = str(obj.get("reason", "")).strip() or "no reason"
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {"action": action, "reason": reason, "confidence": confidence}, "ok"

