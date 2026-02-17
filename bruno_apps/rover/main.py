#!/usr/bin/env python3
"""
Bruno camera rover with camera-only safety and VLM fallback routing.

Key behavior:
- Camera CV is always the hard safety authority.
- VLM is advisory and only used in ambiguous perception states.
- Local llama.cpp server is tried first; remote Gemma fallback is automatic.
"""

import argparse
import base64
import json
import logging
import os
import select
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import termios
import tty


ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.camera.factory import make_camera, read_or_reconnect
from bruno_core.config.env import get_env_bool, get_env_float, get_env_int, get_env_str, load_env
from bruno_core.inference.router import VLMRouter

try:
    from bruno_core.motion.mecanum import MecanumWrapper
except Exception:
    MecanumWrapper = None  # type: ignore


LOG = logging.getLogger("bruno.cam_rover")
ALLOWED_ACTIONS = {"left", "right", "forward", "stop"}


@dataclass
class RoverConfig:
    mode: str = "external"
    forward_speed: int = 35
    turn_speed: int = 35
    turn_rotation: float = 0.45
    frame_width: int = 320
    frame_height: int = 240
    min_motion_area: int = 1800
    center_deadband: float = 0.20
    retry_attempts: int = 3
    retry_delay: float = 2.0
    use_vlm: bool = True
    vlm_provider: str = "auto"  # auto | local | remote
    vlm_local_base: str = "http://127.0.0.1:8080/v1"
    vlm_remote_base: str = "http://localhost:1234/v1"
    vlm_local_model: str = "gemma3"
    vlm_remote_model: str = "gemma3"
    vlm_api_key: str = "lm-studio"
    vlm_local_timeout_ms: int = 800
    vlm_remote_timeout_ms: int = 1600
    vlm_max_latency_ms: int = 1200
    vlm_min_confidence: float = 0.45
    vlm_interval_s: float = 1.5
    vlm_cooldown_s: float = 5.0
    risk_stop_thresh: float = 0.80
    risk_caution_thresh: float = 0.60
    confidence_min: float = 0.45
    uncertain_frame_limit: int = 10
    stop_hold_s: float = 0.8
    search_switch_s: float = 1.5


class LegacyVLMRouter:
    """Routes VLM requests to local first, then remote fallback."""

    def __init__(self, cfg: RoverConfig):
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

        # Auto: local first unless in cooldown.
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
        return self._query_backend(
            name="local",
            api_base=self.cfg.vlm_local_base,
            model=self.cfg.vlm_local_model,
            timeout_ms=self.cfg.vlm_local_timeout_ms,
            frame=frame,
            scene_hint=scene_hint,
        )

    def _query_remote(self, frame, scene_hint: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        return self._query_backend(
            name="remote",
            api_base=self.cfg.vlm_remote_base,
            model=self.cfg.vlm_remote_model,
            timeout_ms=self.cfg.vlm_remote_timeout_ms,
            frame=frame,
            scene_hint=scene_hint,
        )

    def _query_backend(
        self,
        name: str,
        api_base: str,
        model: str,
        timeout_ms: int,
        frame,
        scene_hint: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"provider": name, "reason": "unknown", "latency_ms": 0}
        try:
            payload = self._build_payload(frame, scene_hint, model)
        except Exception as exc:
            meta["reason"] = f"encode_failed:{exc}"
            return None, meta

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.vlm_api_key}",
        }
        url = api_base.rstrip("/") + "/chat/completions"

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=max(0.1, timeout_ms / 1000.0))
            latency_ms = int((time.time() - t0) * 1000)
            meta["latency_ms"] = latency_ms
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            meta["reason"] = f"request_failed:{exc}"
            return None, meta

        action_obj, parse_reason = self._parse_response(data)
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

    def _build_payload(self, frame, scene_hint: str, model: str) -> Dict[str, Any]:
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

    def _parse_response(self, data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
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


class TerminalKeyPoller:
    """Non-blocking single-char key poller for terminal stdin."""

    def __init__(self):
        self._enabled = False
        self._fd = None
        self._old_attrs = None
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._enabled = True
        except Exception as exc:
            LOG.warning("Terminal key polling unavailable: %s", exc)
            self._enabled = False

    def poll(self) -> Optional[str]:
        if not self._enabled or self._fd is None:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            ch = sys.stdin.read(1)
            return ch.lower() if ch else None
        except Exception:
            return None

    def close(self) -> None:
        if not self._enabled or self._fd is None or self._old_attrs is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        except Exception:
            pass


class CameraMotionRover:
    def __init__(self, cfg: RoverConfig, show: bool, max_runtime_s: float):
        self.cfg = cfg
        self.show = show
        self.max_runtime_s = max_runtime_s

        self.camera = make_camera(cfg.mode, cfg.retry_attempts, cfg.retry_delay)
        self.motion = MecanumWrapper(forward_speed=cfg.forward_speed, turn_speed=cfg.turn_speed) if MecanumWrapper else None
        self.vlm = VLMRouter(cfg)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=250,
            varThreshold=28,
            detectShadows=False,
        )

        self.last_good_frame = None
        self.last_cmd = "stop"
        self.search_right = False
        self.last_search_switch = time.time()
        self.stop_latch_until = 0.0
        self.uncertain_frames = 0
        self.turn_history: list[Tuple[float, str]] = []
        self.armed = False
        self.last_key_event = "-"
        self.key_poller = TerminalKeyPoller()

    def run(self) -> None:
        LOG.info("Starting camera rover mode=%s vlm_provider=%s", self.cfg.mode, self.cfg.vlm_provider)
        LOG.info("Controls: y=arm movement, x=immediate stop+disarm, q=quit")
        if not self.camera.open():
            LOG.error("Camera failed to open")
            return

        t0 = time.time()
        try:
            while True:
                key = self.key_poller.poll()
                if key is not None and self._handle_key(key):
                    break

                frame = self._read_frame()
                if frame is None:
                    self._apply_command("stop")
                    time.sleep(0.03)
                    continue

                command, debug = self._decide(frame)
                if not self.armed:
                    debug["manual_gate"] = "disarmed"
                    self._apply_command("stop")
                else:
                    debug["manual_gate"] = "armed"
                    self._apply_command(command)

                if self.show:
                    self._draw_debug(frame, command, debug)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if self.max_runtime_s > 0 and (time.time() - t0) >= self.max_runtime_s:
                    LOG.info("Max runtime reached (%.1fs)", self.max_runtime_s)
                    break

                time.sleep(0.03)

        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_motion()
        self.key_poller.close()
        try:
            self.camera.release()
        except Exception:
            pass
        if self.show:
            cv2.destroyAllWindows()

    def _handle_key(self, key: str) -> bool:
        if key == "y":
            if not self.armed:
                self.armed = True
                self.last_key_event = "arm(y)"
                LOG.info("Manual arm enabled")
            return False
        if key == "x":
            self.armed = False
            self.last_key_event = "stop(x)"
            LOG.warning("Emergency disarm: immediate stop")
            self._apply_command("stop")
            return False
        if key == "q":
            self.last_key_event = "quit(q)"
            LOG.info("Quit requested from terminal")
            return True
        return False

    def _read_frame(self):
        try:
            ok, frame = self.camera.read()
        except Exception:
            ok, frame = False, None

        if ok and frame is not None:
            self.last_good_frame = frame
            return frame

        self.last_good_frame = read_or_reconnect(self.camera, self.last_good_frame)
        return self.last_good_frame

    def _decide(self, frame) -> Tuple[str, Dict[str, Any]]:
        now = time.time()
        if now < self.stop_latch_until:
            return "stop", {"reason": "stop_latch", "risk": 1.0, "confidence": 0.0}

        resized = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self.bg_subtractor.apply(gray)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        bbox = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            best_area = int(cv2.contourArea(c))
            if best_area >= self.cfg.min_motion_area:
                bbox = cv2.boundingRect(c)

        risk = self._estimate_risk(gray)
        confidence = self._estimate_confidence(gray, best_area)

        debug: Dict[str, Any] = {
            "reason": "cv",
            "mask": fg,
            "bbox": bbox,
            "area": best_area,
            "risk": risk,
            "confidence": confidence,
        }

        # Hard camera-only safety gates.
        if risk >= self.cfg.risk_stop_thresh:
            self.stop_latch_until = now + self.cfg.stop_hold_s
            debug["reason"] = f"cv_emergency risk={risk:.2f}"
            return "stop", debug

        if risk >= self.cfg.risk_caution_thresh:
            debug["reason"] = f"cv_caution risk={risk:.2f}"
            return self._search_command(), debug

        unstable = self._is_turn_oscillating(now)
        ambiguous = (bbox is None) or (confidence < self.cfg.confidence_min) or unstable

        if ambiguous:
            self.uncertain_frames += 1
            hint = f"ambiguous confidence={confidence:.2f} area={best_area} unstable={unstable}"
            vlm_action, vlm_meta = self.vlm.decide(resized, hint)
            debug["vlm"] = vlm_meta

            if vlm_action is not None:
                action = vlm_action["action"]
                # VLM can never force forward in borderline confidence.
                if action == "forward" and confidence < (self.cfg.confidence_min + 0.1):
                    debug["reason"] = "vlm_forward_blocked_low_conf"
                    return "stop", debug
                debug["reason"] = f"vlm_{vlm_meta.get('provider')}:{vlm_meta.get('reason')}"
                return action, debug

            # stop then scan policy when uncertain.
            stop_then_scan_frames = max(1, self.cfg.uncertain_frame_limit // 3)
            if self.uncertain_frames <= stop_then_scan_frames:
                debug["reason"] = "uncertain_stop"
                return "stop", debug

            debug["reason"] = "uncertain_scan"
            return self._search_command(), debug

        self.uncertain_frames = 0

        x, y, bw, bh = bbox
        cx = x + (bw / 2.0)
        offset = (cx - (self.cfg.frame_width / 2.0)) / (self.cfg.frame_width / 2.0)
        debug["offset"] = offset

        if offset < -self.cfg.center_deadband:
            self._mark_turn(now, "left")
            return "left", debug
        if offset > self.cfg.center_deadband:
            self._mark_turn(now, "right")
            return "right", debug
        return "forward", debug

    def _estimate_risk(self, gray) -> float:
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 60, 180)

        lower_start = int(h * 0.55)
        center_x1 = int(w * 0.2)
        center_x2 = int(w * 0.8)

        lower_roi = edges[lower_start:h, :]
        center_roi = edges[lower_start:h, center_x1:center_x2]

        lower_occ = float(np.count_nonzero(lower_roi)) / float(max(1, lower_roi.size))
        center_occ = float(np.count_nonzero(center_roi)) / float(max(1, center_roi.size))

        # Camera-only proximity proxy from near-field edge density.
        risk = min(1.0, (0.7 * center_occ + 0.3 * lower_occ) * 9.0)
        return risk

    def _estimate_confidence(self, gray, best_area: int) -> float:
        motion_conf = min(1.0, float(best_area) / float(max(1, self.cfg.min_motion_area * 4)))
        brightness = float(np.mean(gray)) / 255.0
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_conf = min(1.0, sharpness / 500.0)
        return max(0.0, min(1.0, 0.5 * motion_conf + 0.3 * brightness + 0.2 * sharp_conf))

    def _mark_turn(self, now: float, cmd: str) -> None:
        self.turn_history.append((now, cmd))
        # Keep last 2 seconds only.
        self.turn_history = [x for x in self.turn_history if now - x[0] <= 2.0]

    def _is_turn_oscillating(self, now: float) -> bool:
        recent = [cmd for t, cmd in self.turn_history if now - t <= 2.0]
        if len(recent) < 4:
            return False
        flips = 0
        for i in range(1, len(recent)):
            if recent[i] != recent[i - 1]:
                flips += 1
        return flips >= 3

    def _search_command(self) -> str:
        now = time.time()
        if now - self.last_search_switch >= self.cfg.search_switch_s:
            self.search_right = not self.search_right
            self.last_search_switch = now
        cmd = "right" if self.search_right else "left"
        self._mark_turn(now, cmd)
        return cmd

    def _apply_command(self, command: str) -> None:
        if command == self.last_cmd:
            return

        self.last_cmd = command
        LOG.info("Command: %s", command)

        if not self.motion:
            return

        try:
            if command == "forward":
                self.motion.set_velocity(self.cfg.forward_speed, 90, 0)
            elif command == "left":
                self.motion.set_velocity(0, 90, -self.cfg.turn_rotation)
            elif command == "right":
                self.motion.set_velocity(0, 90, self.cfg.turn_rotation)
            else:
                self.motion.stop()
        except Exception:
            pass

    def _stop_motion(self) -> None:
        if not self.motion:
            return
        try:
            self.motion.stop()
        except Exception:
            pass

    def _draw_debug(self, frame, command: str, debug: Dict[str, Any]) -> None:
        view = frame.copy()
        h, w = view.shape[:2]

        cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        dead = int((self.cfg.center_deadband * w) / 2.0)
        cv2.line(view, (w // 2 - dead, 0), (w // 2 - dead, h), (100, 100, 255), 1)
        cv2.line(view, (w // 2 + dead, 0), (w // 2 + dead, h), (100, 100, 255), 1)

        bbox = debug.get("bbox")
        if bbox is not None:
            sx = w / float(self.cfg.frame_width)
            sy = h / float(self.cfg.frame_height)
            x, y, bw, bh = bbox
            x1 = int(x * sx)
            y1 = int(y * sy)
            x2 = int((x + bw) * sx)
            y2 = int((y + bh) * sy)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

        risk = float(debug.get("risk", 0.0))
        confidence = float(debug.get("confidence", 0.0))
        manual_gate = str(debug.get("manual_gate", "unknown"))
        effective_cmd = command if self.armed else "stop"

        cv2.putText(view, f"cmd: {effective_cmd}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, f"risk: {risk:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.putText(view, f"conf: {confidence:.2f}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(view, f"state: {debug.get('reason', 'unknown')}", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 2)
        cv2.putText(view, f"manual: {manual_gate} key={self.last_key_event}", (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 180), 1)

        vlm = debug.get("vlm") or {}
        if vlm:
            cv2.putText(
                view,
                f"vlm: {vlm.get('provider','none')} {vlm.get('reason','')} {vlm.get('latency_ms',0)}ms",
                (12, 166),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 255),
                1,
            )

        cv2.imshow("Bruno Camera Rover", view)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bruno camera rover with local/remote VLM fallback")
    parser.add_argument("--mode", choices=["builtin", "external"], default=get_env_str("CAM_MODE"))
    parser.add_argument("--show", action="store_true", help="Show debug camera window")
    parser.add_argument("--max-runtime", type=float, default=0.0, help="Stop after N seconds (0 = run forever)")

    parser.add_argument("--forward-speed", type=int, default=get_env_int("BRUNO_ROVER_FORWARD_SPEED"))
    parser.add_argument("--turn-speed", type=int, default=get_env_int("BRUNO_ROVER_TURN_SPEED"))
    parser.add_argument("--turn-rotation", type=float, default=get_env_float("BRUNO_ROVER_TURN_ROT"))
    parser.add_argument("--min-motion-area", type=int, default=get_env_int("BRUNO_ROVER_MIN_MOTION_AREA"))
    parser.add_argument("--center-deadband", type=float, default=get_env_float("BRUNO_ROVER_CENTER_DEADBAND"))

    parser.add_argument("--risk-stop-thresh", type=float, default=get_env_float("BRUNO_ROVER_RISK_STOP"))
    parser.add_argument("--risk-caution-thresh", type=float, default=get_env_float("BRUNO_ROVER_RISK_CAUTION"))
    parser.add_argument("--confidence-min", type=float, default=get_env_float("BRUNO_ROVER_CONFIDENCE_MIN"))
    parser.add_argument("--stop-hold-s", type=float, default=get_env_float("BRUNO_ROVER_STOP_HOLD_S"))
    parser.add_argument("--uncertain-frame-limit", type=int, default=get_env_int("BRUNO_ROVER_UNCERTAIN_FRAME_LIMIT"))
    parser.add_argument("--search-switch-s", type=float, default=get_env_float("BRUNO_ROVER_SEARCH_SWITCH_S"))

    parser.add_argument("--use-vlm", dest="use_vlm", action="store_true", help="Enable advisory VLM (default)")
    parser.add_argument("--no-vlm", dest="use_vlm", action="store_false", help="Disable advisory VLM")
    parser.set_defaults(use_vlm=get_env_bool("BRUNO_USE_VLM"))
    parser.add_argument("--vlm-provider", choices=["auto", "local", "remote"], default=get_env_str("BRUNO_VLM_PROVIDER"))
    parser.add_argument("--vlm-local-base", default=get_env_str("BRUNO_VLM_LOCAL_BASE"))
    parser.add_argument("--vlm-remote-base", default=get_env_str("BRUNO_VLM_REMOTE_BASE", get_env_str("BRUNO_VLM_API_BASE")))
    parser.add_argument("--vlm-local-model", default=get_env_str("BRUNO_VLM_LOCAL_MODEL"))
    parser.add_argument("--vlm-remote-model", default=get_env_str("BRUNO_VLM_REMOTE_MODEL", get_env_str("BRUNO_VLM_MODEL")))
    parser.add_argument("--vlm-api-key", default=get_env_str("BRUNO_VLM_API_KEY"))
    parser.add_argument("--vlm-local-timeout-ms", type=int, default=get_env_int("BRUNO_VLM_LOCAL_TIMEOUT_MS"))
    parser.add_argument("--vlm-remote-timeout-ms", type=int, default=get_env_int("BRUNO_VLM_REMOTE_TIMEOUT_MS"))
    parser.add_argument("--vlm-max-latency-ms", type=int, default=get_env_int("BRUNO_VLM_MAX_LATENCY_MS"))
    parser.add_argument("--vlm-min-confidence", type=float, default=get_env_float("BRUNO_VLM_MIN_CONFIDENCE"))
    parser.add_argument("--vlm-interval", type=float, default=get_env_float("BRUNO_VLM_INTERVAL_SEC"))
    parser.add_argument("--vlm-cooldown-s", type=float, default=get_env_float("BRUNO_VLM_COOLDOWN_S"))

    return parser.parse_args()


def main() -> None:
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = RoverConfig(
        mode=args.mode,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        turn_rotation=args.turn_rotation,
        min_motion_area=args.min_motion_area,
        center_deadband=args.center_deadband,
        use_vlm=args.use_vlm,
        vlm_provider=args.vlm_provider,
        vlm_local_base=args.vlm_local_base,
        vlm_remote_base=args.vlm_remote_base,
        vlm_local_model=args.vlm_local_model,
        vlm_remote_model=args.vlm_remote_model,
        vlm_api_key=args.vlm_api_key,
        vlm_local_timeout_ms=args.vlm_local_timeout_ms,
        vlm_remote_timeout_ms=args.vlm_remote_timeout_ms,
        vlm_max_latency_ms=args.vlm_max_latency_ms,
        vlm_min_confidence=args.vlm_min_confidence,
        vlm_interval_s=args.vlm_interval,
        vlm_cooldown_s=args.vlm_cooldown_s,
        risk_stop_thresh=args.risk_stop_thresh,
        risk_caution_thresh=args.risk_caution_thresh,
        confidence_min=args.confidence_min,
        stop_hold_s=args.stop_hold_s,
        uncertain_frame_limit=args.uncertain_frame_limit,
        search_switch_s=args.search_switch_s,
    )

    rover = CameraMotionRover(cfg=cfg, show=args.show, max_runtime_s=args.max_runtime)
    rover.run()


if __name__ == "__main__":
    main()
