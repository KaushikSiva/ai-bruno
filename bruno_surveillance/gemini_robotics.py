"""
Gemini Robotics policy integration (optional).

This module formulates a simple action schema for Bruno and asks
Google's Gemini multimodal API to propose the next action based on
the latest camera snapshot and caption. It returns a small JSON
object with fields: {action, duration_s, speed, reason}.

Two backends are supported:
- Official Python SDK (google-genai): set GEMINI_USE_SDK=1
- REST fallback (no SDK): default when SDK is unavailable/disabled

Env:
- GEMINI_API_KEY / GOOGLE_API_KEY: required
- GEMINI_MODEL:    model name (default: gemini-1.5-flash)
- GEMINI_API_BASE: REST base URL (default: https://generativelanguage.googleapis.com)
- GEMINI_USE_SDK:  "1" to use google-genai client if installed

The SDK path also supports tool usage (e.g., code execution) per
Google’s robotics examples.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

from utils import LOG

_SDK_AVAILABLE = False
try:
    # New official SDK: pip install google-genai
    from goog import genai as _genai
    from google.genai import types as _gtypes
    _SDK_AVAILABLE = True
except Exception:
    _SDK_AVAILABLE = False


ALLOWED_ACTIONS = {"forward", "reverse", "left", "right", "stop"}


def _b64_image(path: str) -> tuple[str, str] | tuple[None, None]:
    try:
        with open(path, "rb") as f:
            b = f.read()
        # Heuristic mime type
        ext = (os.path.splitext(path)[1] or "").lower()
        mime = "image/jpeg"
        if ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        return base64.b64encode(b).decode("ascii"), mime
    except Exception as e:
        LOG.warning(f"gemini_robotics: failed to read image {path}: {e}")
        return None, None


def _strip_code_fences(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        # drop opening fence + optional language
        nl = t.find("\n")
        if nl > 0:
            t = t[nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _safe_json_parse(txt: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(txt)
    except Exception:
        return None


def _coerce_action(obj: Dict[str, Any]) -> Dict[str, Any]:
    action = str(obj.get("action", "stop")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = "stop"
    try:
        duration_s = float(obj.get("duration_s", 0.5))
    except Exception:
        duration_s = 0.5
    # Clamp duration to a safe window
    if duration_s < 0:
        duration_s = 0.0
    if duration_s > 2.0:
        duration_s = 2.0
    try:
        speed = int(obj.get("speed", 40))
    except Exception:
        speed = 40
    if speed < 0:
        speed = 0
    if speed > 100:
        speed = 100
    reason = str(obj.get("reason", "")).strip()
    return {"action": action, "duration_s": duration_s, "speed": speed, "reason": reason}


@dataclass
class GeminiPolicy:
    api_key: Optional[str] = None
    model: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    api_base: str = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    target_object: Optional[str] = None
    target_mode: str = "policy"  # 'policy' (LLM policy) or 'bbox' (deterministic)
    _scan_right: bool = False

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # Optional goal: approach a specific object (e.g., "water bottle")
        self.target_object = (
            os.environ.get("GEMINI_TARGET_OBJECT")
            or os.environ.get("TARGET_OBJECT")
            or self.target_object
        )
        # Target mode: 'policy' or 'bbox' (aka deterministic)
        tm = (os.environ.get("GEMINI_TARGET_MODE") or self.target_mode or "policy").strip().lower()
        if tm == "deterministic":
            tm = "bbox"
        if tm not in ("policy", "bbox"):
            tm = "policy"
        self.target_mode = tm

    def enabled(self) -> bool:
        return bool(self.api_key)

    def decide_action(
        self,
        image_path: str,
        caption: str,
        last_distance_cm: Optional[float] = None,
        ultra_caution_cm: Optional[float] = None,
        ultra_danger_cm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ask Gemini to propose the next action based on the current observation.

        Returns a normalized dict with keys: action, duration_s, speed, reason.
        Fallbacks to a safe STOP when the API is not configured or fails.
        """
        if not self.enabled():
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "Gemini API key not set"}
        # Deterministic target-follow (bbox) if requested and a target is set
        if (self.target_object or "").strip() and self.target_mode == "bbox":
            act = self._decide_with_bbox(
                image_path,
                caption,
                last_distance_cm,
                ultra_caution_cm,
                ultra_danger_cm,
            )
            if act:
                return act
        # SDK takes precedence when requested and available
        use_sdk = (os.environ.get("GEMINI_USE_SDK", "").strip().lower() in ("1", "true", "yes")) and _SDK_AVAILABLE
        if use_sdk:
            return self._decide_with_sdk(image_path, caption, last_distance_cm, ultra_caution_cm, ultra_danger_cm)
        else:
            return self._decide_with_rest(image_path, caption, last_distance_cm, ultra_caution_cm, ultra_danger_cm)

    # ----- Backends -----
    def _decide_with_sdk(
        self,
        image_path: str,
        caption: str,
        last_distance_cm: Optional[float],
        ultra_caution_cm: Optional[float],
        ultra_danger_cm: Optional[float],
    ) -> Dict[str, Any]:
        if not _SDK_AVAILABLE:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "sdk not installed"}
        try:
            client = _genai.Client(api_key=self.api_key)
        except Exception as e:
            LOG.warning(f"gemini sdk client failed: {e}")
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"sdk client error: {e}"}

        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"img read error: {e}"}

        # Build image part
        # New SDK supports bytes part via types.Part.from_bytes
        try:
            img_part = _gtypes.Part.from_bytes(img_bytes, mime_type=_guess_mime(image_path))
        except Exception:
            # Fallback to older style
            b64, mime = _b64_image(image_path)
            if not b64:
                return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "could not read image"}
            img_part = _gtypes.Part(inline_data=_gtypes.Blob(mime_type=mime, data=b64))

        # Build prompts with optional target goal
        goal = (self.target_object or '').strip()
        base_task = (
            "You control a small wheeled robot named Bruno with primitive actions: "
            "forward, reverse, left, right, stop. You receive one RGB snapshot and a short caption. "
        )
        if goal:
            base_task += (
                f"Your goal is to approach the target object: '{goal}'. "
                "If the target is not visible, suggest a short scan (left/right). "
                "If visible and off-center, steer to center it. If centered and safe, move forward briefly. "
            )
        else:
            base_task += "Propose the next immediate action for smooth navigation while avoiding obstacles. "
        system_prompt = base_task + "Keep motions short and safe. Return ONLY minified JSON per schema."

        safety_ctx = []
        if ultra_caution_cm is not None and ultra_danger_cm is not None:
            safety_ctx.append(
                f"Safety thresholds: caution <= {ultra_caution_cm:.1f} cm; danger <= {ultra_danger_cm:.1f} cm."
            )
        if last_distance_cm is not None:
            safety_ctx.append(f"Ultrasonic reading: {last_distance_cm:.1f} cm.")
        ctx_text = ("\n".join(safety_ctx)) if safety_ctx else ""

        user_prompt = (
            f"Scene caption: {caption}\n{ctx_text}\n"
            "Output schema: {\"action\": [forward|reverse|left|right|stop], "
            "\"duration_s\": float 0..2, \"speed\": int 0..100, \"reason\": string}. "
            "Return JSON only."
        )

        cfg = _gtypes.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=200,
            # Enable code execution tool as per robotics examples; model may use it to reason/zoom.
            tools=[_gtypes.Tool(code_execution=_gtypes.ToolCodeExecution())],
        )
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[img_part, user_prompt, system_prompt],
                config=cfg,
            )
        except Exception as e:
            LOG.warning(f"gemini sdk call failed: {e}")
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"sdk call error: {e}"}

        # Collect any text parts (and ignore tool code/results here)
        txt_chunks = []
        try:
            cand = resp.candidates[0]
            for part in getattr(cand.content, 'parts', []) or []:
                if getattr(part, 'text', None):
                    txt_chunks.append(part.text)
        except Exception:
            pass
        txt = _strip_code_fences("\n".join(txt_chunks).strip()) if txt_chunks else ""
        if not txt:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "empty sdk response"}
        obj = _safe_json_parse(txt) or _safe_json_parse(_extract_json_substring(txt))
        if not obj or not isinstance(obj, dict):
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "unparsable sdk response"}
        return _coerce_action(obj)

    def _decide_with_rest(
        self,
        image_path: str,
        caption: str,
        last_distance_cm: Optional[float],
        ultra_caution_cm: Optional[float],
        ultra_danger_cm: Optional[float],
    ) -> Dict[str, Any]:
        b64, mime = _b64_image(image_path)
        if not b64:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "could not read image"}

        # REST prompt mirrors SDK behavior
        goal = (self.target_object or '').strip()
        prompt = (
            "You control a small wheeled robot named Bruno with primitive actions: "
            "forward, reverse, left, right, stop. You receive a single RGB snapshot "
            "and a short caption of the scene. "
        )
        if goal:
            prompt += (
                f"Your goal is to approach the target object: '{goal}'. "
                "If the target is not visible, suggest a short scan (left/right). "
                "If visible and off-center, steer to center it; if centered and safe, move forward briefly. "
            )
        else:
            prompt += "Propose the next immediate action for smooth navigation while avoiding obstacles. "
        prompt += (
            "Keep movements short.\n\n"
            "Output ONLY compact minified JSON with this schema: "
            "{\"action\": one of [forward,reverse,left,right,stop], "
            "\"duration_s\": float seconds (0..2), \"speed\": int 0..100, "
            "\"reason\": short explanation}. No extra text."
        )

        # Include safety context if available
        safety_ctx = ""
        if ultra_caution_cm is not None and ultra_danger_cm is not None:
            safety_ctx = (
                f"\nSafety thresholds: caution <= {ultra_caution_cm:.1f} cm, "
                f"danger <= {ultra_danger_cm:.1f} cm. "
                "Avoid proposing forward if likely too close."
            )
        if last_distance_cm is not None:
            safety_ctx += f"\nLatest ultrasonic distance (if available): {last_distance_cm:.1f} cm."

        user_text = (
            f"Scene caption: {caption}\n"
            f"Robot: Bruno with mecanum chassis (can pivot left/right).\n"
            f"Environment: indoor home/office.\n"
            f"Constraints: avoid obstacles and people. Short, safe motions only.{safety_ctx}\n"
            f"Return JSON only."
        )

        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"text": user_text},
                        {
                            "inline_data": {
                                "mime_type": mime,
                                "data": b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
        }

        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            txt = _extract_text_from_rest(data)
            if not txt:
                return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "empty response"}
            txt = _strip_code_fences(txt)
            obj = _safe_json_parse(txt) or _safe_json_parse(_extract_json_substring(txt))
            if not obj or not isinstance(obj, dict):
                return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": "unparsable response"}
            return _coerce_action(obj)
        except Exception as e:
            LOG.warning(f"Gemini policy call failed: {e}")
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"error: {e}"}

    def _decide_with_bbox(
        self,
        image_path: str,
        caption: str,
        last_distance_cm: Optional[float],
        ultra_caution_cm: Optional[float],
        ultra_danger_cm: Optional[float],
    ) -> Dict[str, Any]:
        """
        Deterministic centering/approach behavior using a target bounding box.
        1) Ask Gemini to detect a bbox for the target object (or return null).
        2) If not found: suggest a short scanning turn (alternating).
        3) If found: steer left/right to center; if centered and safe, move forward.
        """
        target = (self.target_object or "").strip() or "target object"
        bbox = self._detect_target_bbox(image_path, target)
        if not bbox:
            # Alternate scan left/right to search
            self._scan_right = not self._scan_right
            direction = "right" if self._scan_right else "left"
            return {"action": direction, "duration_s": 0.5, "speed": 40, "reason": f"scan for {target}"}

        # Compute action from bbox + safety context
        return self._compute_action_from_bbox(
            image_path, bbox, last_distance_cm, ultra_caution_cm, ultra_danger_cm, target
        )

    def _compute_action_from_bbox(
        self,
        image_path: str,
        bbox: Dict[str, int],
        last_distance_cm: Optional[float],
        ultra_caution_cm: Optional[float],
        ultra_danger_cm: Optional[float],
        target: str,
    ) -> Dict[str, Any]:
        w_img, h_img = self._image_size(image_path)
        x, y, w, h = (
            int(bbox.get("x", 0)),
            int(bbox.get("y", 0)),
            int(bbox.get("w", 0)),
            int(bbox.get("h", 0)),
        )
        # Center offset
        if w_img > 0:
            cx = x + (w / 2.0)
            offset = (cx / float(w_img)) - 0.5  # negative => left, positive => right
        else:
            offset = 0.0
        # Size ratio as crude proximity cue
        size_ratio = (w / float(w_img)) if w_img > 0 else 0.0

        # Safety gating
        if last_distance_cm is not None and ultra_danger_cm is not None and last_distance_cm <= ultra_danger_cm:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"danger zone ({last_distance_cm:.1f} cm)"}

        # Steering thresholds
        deadband = 0.1  # within 10% of image center => considered centered
        turn_time = 0.35 if abs(offset) < 0.2 else 0.5
        if offset < -deadband:
            return {"action": "left", "duration_s": turn_time, "speed": 40, "reason": f"center {target}: offset left {offset:.2f}"}
        if offset > deadband:
            return {"action": "right", "duration_s": turn_time, "speed": 40, "reason": f"center {target}: offset right {offset:.2f}"}

        # Centered: consider approaching
        near_ratio = 0.35
        if size_ratio >= near_ratio:
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"{target} close (ratio {size_ratio:.2f})"}

        if last_distance_cm is not None and ultra_caution_cm is not None and last_distance_cm <= ultra_caution_cm:
            # In caution zone: small adjustments only
            return {"action": "stop", "duration_s": 0.0, "speed": 0, "reason": f"caution zone ({last_distance_cm:.1f} cm)"}

        return {"action": "forward", "duration_s": 0.6, "speed": 40, "reason": f"approach {target} (ratio {size_ratio:.2f})"}

    def _image_size(self, path: str) -> tuple[int, int]:
        try:
            if Image is None:
                return (0, 0)
            with Image.open(path) as im:
                im.load()
                return int(im.size[0]), int(im.size[1])
        except Exception:
            return (0, 0)

    def _detect_target_bbox(self, image_path: str, target: str) -> Optional[Dict[str, int]]:
        use_sdk = (os.environ.get("GEMINI_USE_SDK", "").strip().lower() in ("1", "true", "yes")) and _SDK_AVAILABLE
        if use_sdk:
            return self._detect_with_sdk(image_path, target)
        return self._detect_with_rest(image_path, target)

    def _detect_with_sdk(self, image_path: str, target: str) -> Optional[Dict[str, int]]:
        if not _SDK_AVAILABLE:
            return None
        try:
            client = _genai.Client(api_key=self.api_key)
        except Exception:
            return None
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except Exception:
            return None
        try:
            img_part = _gtypes.Part.from_bytes(img_bytes, mime_type=_guess_mime(image_path))
        except Exception:
            b64, mime = _b64_image(image_path)
            if not b64:
                return None
            img_part = _gtypes.Part(inline_data=_gtypes.Blob(mime_type=mime, data=b64))

        system = (
            "You are a precise vision detector. Reply ONLY with minified JSON. "
            f"Detect a tight bounding box around the most prominent '{target}'. "
            "If not present, return {\"bbox\": null}. JSON schema: {\"bbox\":{\"x\":int,\"y\":int,\"w\":int,\"h\":int}}."
        )
        cfg = _gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=120)
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[img_part, system],
                config=cfg,
            )
        except Exception:
            return None
        txt_chunks = []
        try:
            cand = resp.candidates[0]
            for part in getattr(cand.content, 'parts', []) or []:
                if getattr(part, 'text', None):
                    txt_chunks.append(part.text)
        except Exception:
            pass
        txt = _strip_code_fences("\n".join(txt_chunks).strip()) if txt_chunks else ""
        obj = _safe_json_parse(txt) or _safe_json_parse(_extract_json_substring(txt))
        return _coerce_bbox(obj)

    def _detect_with_rest(self, image_path: str, target: str) -> Optional[Dict[str, int]]:
        b64, mime = _b64_image(image_path)
        if not b64:
            return None
        system = (
            "You are a precise vision detector. Reply ONLY with minified JSON. "
            f"Detect a tight bounding box around the most prominent '{target}'. "
            "If not present, return {\"bbox\": null}. JSON schema: {\"bbox\":{\"x\":int,\"y\":int,\"w\":int,\"h\":int}}."
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [
                    {"text": system},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]}
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 120},
        }
        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            txt = _extract_text_from_rest(data)
            txt = _strip_code_fences(txt)
            obj = _safe_json_parse(txt) or _safe_json_parse(_extract_json_substring(txt))
            return _coerce_bbox(obj)
        except Exception:
            return None


# ----- Helpers -----
def _guess_mime(path: str) -> str:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def _extract_json_substring(txt: str) -> str:
    start = txt.find("{")
    end = txt.rfind("}")
    if 0 <= start < end:
        return txt[start : end + 1]
    return ""


def _extract_text_from_rest(data: Dict[str, Any]) -> str:
    txt = ""
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        try:
            # Alternative shapes
            cands = data.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts") or []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        txt += (p["text"] or "") + "\n"
        except Exception:
            pass
    return txt.strip()


def _coerce_bbox(obj: Any) -> Optional[Dict[str, int]]:
    try:
        if not isinstance(obj, dict):
            return None
        bbox = obj.get("bbox")
        if not isinstance(bbox, dict):
            return None
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", 0))
        h = int(bbox.get("h", 0))
        if w <= 0 or h <= 0:
            return None
        return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        return None
