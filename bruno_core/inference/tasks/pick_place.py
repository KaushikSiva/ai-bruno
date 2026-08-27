"""
VLM task: turn a natural-language instruction into a grounded pick-place plan.

The model is asked only for *semantics* — which object to grab and roughly
where to put it — never for metric coordinates. Depth from a single frame is
unreliable and the 4-DOF arm has a small fixed workspace, so grasp geometry is
owned by CV (HSV/YOLO centroid + the focal-length distance estimator) while the
VLM decides what "the blue bottle on the left" refers to.
"""

import base64
import json
from typing import Any, Dict, Optional, Tuple

import cv2

# Colors the CV layer can actually track (keys of detection.color_detection).
ALLOWED_COLORS = {"clear", "blue", "green", "any"}
ALLOWED_PLACEMENTS = {"left", "right", "ahead", "home"}

PLAN_SCHEMA = (
    '{"target_description":"<what you see>","target_color":"<clear|blue|green|any>",'
    '"place":"<left|right|ahead|home>","confidence":<0.0-1.0>}'
)

# Small VLMs copy placeholders verbatim, so show a filled-in answer too. Without
# this, LFM2-VL-450M returns the schema's own text as the description and 0.0 as
# the confidence, while still getting color and place right.
PLAN_EXAMPLE = (
    '{"target_description":"blue plastic bottle on the table","target_color":"blue",'
    '"place":"left","confidence":0.9}'
)


def _data_url(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_plan_payload(frame, instruction: str, model: str) -> Dict[str, Any]:
    """Ask the VLM which visible object the instruction refers to."""
    prompt = (
        "You are the perception planner for a small robot arm on a rover. "
        "From the camera frame and the user instruction, identify the single object "
        "to pick up and where to put it. "
        "target_color must be the dominant color of the object and one of: clear, blue, green, any. "
        "place must be one of: left, right, ahead, home. "
        "target_description must describe the object you actually see, in your own "
        "words — never copy the placeholder text. "
        "confidence is how sure you are that the object is present: use 0.8 or higher "
        "when you can see it clearly, and 0.0 only if nothing in the frame matches. "
        f"Return strict minified JSON only with schema {PLAN_SCHEMA}. "
        f"Example of a well-formed answer: {PLAN_EXAMPLE}. "
        f"Instruction: {instruction}"
    )
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 160,
        "messages": [
            {"role": "system", "content": "Return only minified JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(frame)}},
                ],
            },
        ],
    }


def parse_plan_response(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse a plan response. Returns (plan, reason)."""
    obj, reason = _parse_json_content(data)
    if obj is None:
        return None, reason

    color = str(obj.get("target_color", "any")).strip().lower()
    if color not in ALLOWED_COLORS:
        color = "any"

    place = str(obj.get("place", "home")).strip().lower()
    if place not in ALLOWED_PLACEMENTS:
        place = "home"

    description = str(obj.get("target_description", "")).strip()
    if not description or _is_schema_echo(description):
        # Only used for logging and the grasp-check prompt — CV grasps on color,
        # so a copied placeholder is not worth failing the whole plan over.
        description = f"{color} object"

    return (
        {
            "target_description": description,
            "target_color": color,
            "place": place,
            "confidence": _clamp_confidence(obj.get("confidence")),
        },
        "ok",
    )


def build_grasp_check_payload(frame, target_description: str, model: str) -> Dict[str, Any]:
    """Ask the VLM whether the gripper is actually holding the target."""
    prompt = (
        "The robot arm just attempted to grasp an object and lifted it. "
        f"Looking at the frame, is the gripper holding the {target_description}? "
        'Return strict minified JSON only with schema {"holding":true,"confidence":0.0}.'
    )
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 60,
        "messages": [
            {"role": "system", "content": "Return only minified JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(frame)}},
                ],
            },
        ],
    }


def parse_grasp_check_response(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    obj, reason = _parse_json_content(data)
    if obj is None:
        return None, reason
    holding = obj.get("holding")
    if isinstance(holding, str):
        holding = holding.strip().lower() in ("true", "yes", "1")
    return {"holding": bool(holding), "confidence": _clamp_confidence(obj.get("confidence"))}, "ok"


def _parse_json_content(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract and parse the JSON object from a chat completion, tolerating fences."""
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
    return obj, "ok"


def _is_schema_echo(description: str) -> bool:
    """True if the model copied a placeholder instead of describing the scene."""
    text = description.strip().lower().strip("<>")
    return "|" in text or text in ("short", "what you see", "target_description")


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.0
    return max(0.0, min(1.0, confidence))
