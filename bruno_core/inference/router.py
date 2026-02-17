import time
from typing import Any, Dict, Optional, Tuple

import requests

from bruno_core.inference.providers.openai_compat import post_chat_completion
from bruno_core.inference.tasks.navigation import build_nav_payload, parse_nav_response


class VLMRouter:
    """Routes VLM requests to local first, then remote fallback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_query_t = 0.0
        self.local_degraded_until = 0.0

    def decide(self, frame, scene_hint: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        meta = {"provider": "none", "reason": "vlm disabled", "latency_ms": 0}
        if not self.cfg.use_vlm:
            return None, meta
        now = time.time()
        if now - self.last_query_t < self.cfg.vlm_interval_s:
            meta["reason"] = "throttled"
            return None, meta
        self.last_query_t = now
        provider = self.cfg.vlm_provider.strip().lower()
        if provider not in {"auto", "local", "remote"}:
            provider = "auto"
        if provider == "local":
            return self._query_local(frame, scene_hint)
        if provider == "remote":
            return self._query_remote(frame, scene_hint)
        if now >= self.local_degraded_until:
            local_decision, local_meta = self._query_local(frame, scene_hint)
            if local_decision is not None:
                return local_decision, local_meta
            self.local_degraded_until = now + self.cfg.vlm_cooldown_s
            remote_decision, remote_meta = self._query_remote(frame, scene_hint)
            if remote_decision is not None:
                remote_meta["reason"] = f"local_failed:{local_meta.get('reason', 'unknown')}"
            return remote_decision, remote_meta
        remote_decision, remote_meta = self._query_remote(frame, scene_hint)
        if remote_decision is not None:
            remote_meta["reason"] = "local_cooldown"
        return remote_decision, remote_meta

    def _query_local(self, frame, scene_hint: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        return self._query_backend("local", self.cfg.vlm_local_base, self.cfg.vlm_local_model, self.cfg.vlm_local_timeout_ms, frame, scene_hint)

    def _query_remote(self, frame, scene_hint: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        return self._query_backend("remote", self.cfg.vlm_remote_base, self.cfg.vlm_remote_model, self.cfg.vlm_remote_timeout_ms, frame, scene_hint)

    def _query_backend(self, name: str, api_base: str, model: str, timeout_ms: int, frame, scene_hint: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"provider": name, "reason": "unknown", "latency_ms": 0}
        try:
            payload = build_nav_payload(frame, scene_hint, model)
        except Exception as exc:
            meta["reason"] = f"encode_failed:{exc}"
            return None, meta
        t0 = time.time()
        try:
            data = post_chat_completion(
                api_base=api_base,
                api_key=self.cfg.vlm_api_key,
                payload=payload,
                timeout_sec=max(0.1, timeout_ms / 1000.0),
            )
            meta["latency_ms"] = int((time.time() - t0) * 1000)
        except Exception as exc:
            meta["reason"] = f"request_failed:{exc}"
            return None, meta
        action_obj, parse_reason = parse_nav_response(data)
        if action_obj is None:
            meta["reason"] = parse_reason
            return None, meta
        conf = float(action_obj.get("confidence", 0.0))
        if conf < self.cfg.vlm_min_confidence:
            meta["reason"] = f"low_confidence:{conf:.2f}"
            return None, meta
        if int(meta["latency_ms"]) > self.cfg.vlm_max_latency_ms:
            meta["reason"] = f"slow_response:{meta['latency_ms']}ms"
            return None, meta
        meta["reason"] = "ok"
        return action_obj, meta

