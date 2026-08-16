#!/usr/bin/env python3
"""
Bruno prompt-driven pick and place.

Give it a natural-language instruction ("pick up the blue bottle and put it on
the left") and it plans once with the VLM, then executes a CV-closed loop:

    PLAN -> SEARCH -> APPROACH -> ALIGN -> GRASP -> VERIFY -> PLACE -> DONE

Division of labor, mirroring the rover app's safety model:
- VLM decides *semantics* (which object, where to put it) — one call up front.
- CV owns *geometry* (centroid, standoff distance) and is the hard authority.
- Ultrasonic stops the base regardless of what either says.

The target is a plastic bottle. Detection is YOLO first (COCO class "bottle",
the only thing that sees clear PET, which HSV cannot — it shows whatever is
behind it) with the HSV blob detector as fallback for a colored or tape-wrapped
bottle. Distance falls back to the body width once the bottle is too close to
fit in frame, and the arm closes in sideways rather than descending onto a
bottle taller than its own approach clearance.

YOLO needs `pip install ultralytics` on the robot; without it the app logs a
warning and runs HSV-only.

Run on the robot:

    python3 bruno_apps/pick_place/main.py --prompt "pick up the blue bottle"
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.camera.factory import make_camera, read_or_reconnect
from bruno_core.config.env import get_env_float, get_env_int, get_env_str, load_env
from bruno_core.inference.providers.openai_compat import post_chat_completion
from bruno_core.inference.tasks.pick_place import (
    ALLOWED_COLORS,
    ALLOWED_PLACEMENTS,
    build_grasp_check_payload,
    build_plan_payload,
    parse_grasp_check_response,
    parse_plan_response,
)
from bruno_core.logging.setup import LOG
from bruno_core.manipulation.arm import ArmConfig, ArmController

try:
    from bruno_core.motion.mecanum import MecanumWrapper
except Exception:
    MecanumWrapper = None  # type: ignore

try:
    from bruno_core.sensors.ultrasonic import UltrasonicRGB
except Exception:
    UltrasonicRGB = None  # type: ignore


CONFIG_PATH = os.path.join(REPO_ROOT, "config", "bruno_config.json")


class State(Enum):
    PLAN = "plan"
    SEARCH = "search"
    APPROACH = "approach"
    ALIGN = "align"
    GRASP = "grasp"
    VERIFY = "verify"
    PLACE = "place"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PickPlaceConfig:
    mode: str = "builtin"
    frame_width: int = 640
    frame_height: int = 480
    # CV target selection
    min_area: int = 500
    max_area: int = 150000
    # A bottle is tall and narrow; the ratio is what separates it from a wall,
    # a table edge, or a patch of carpet that happens to fall in the HSV range.
    min_aspect_ratio: float = 1.2
    max_aspect_ratio: float = 4.5
    center_deadband: float = 0.12
    # YOLO target selection (class-based; the only thing that sees clear PET)
    use_yolo: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_classes: Tuple[str, ...] = ("bottle",)
    yolo_imgsz: int = 320
    yolo_min_area: int = 300
    # approach geometry (cm)
    focal_length: float = 500.0
    real_object_height: float = 20.0
    real_object_width: float = 7.0
    grasp_standoff_cm: float = 18.0
    standoff_tolerance_cm: float = 3.0
    # base motion
    forward_speed: int = 25
    turn_rotation: float = 0.35
    nudge_s: float = 0.25
    search_turn_s: float = 0.4
    # safety
    ultra_danger_cm: float = 12.0
    # VLM
    vlm_base: str = "http://127.0.0.1:8081/v1"
    vlm_model: str = "gemma3"
    vlm_api_key: str = "lm-studio"
    vlm_timeout_ms: int = 8000
    vlm_min_confidence: float = 0.35
    # Set by --target-color/--place to skip the VLM entirely: lets the CV +
    # arm pipeline be tested without depending on the model's judgement.
    plan_override: Optional[Dict[str, Any]] = None
    # loop limits
    search_timeout_s: float = 25.0
    approach_timeout_s: float = 45.0
    grasp_retries: int = 2


def load_json_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        LOG.warning("Could not read %s: %s", CONFIG_PATH, exc)
        return {}


def estimate_distance_cm(cfg: "PickPlaceConfig", target: Dict[str, Any]) -> Optional[float]:
    """Pinhole estimate: distance = focal_length * real_size / pixel_size.

    Height is the better cue while the whole bottle is visible. Once it is
    clipped by the frame edge, the width of the body is the only dimension
    still measured end-to-end, so switch to it rather than reporting a distance
    that grows as the robot closes in.
    """
    if target.get("clipped"):
        pixel_width = int(target.get("pixel_width", 0))
        if pixel_width <= 0:
            return None
        return (cfg.focal_length * cfg.real_object_width) / float(pixel_width)
    pixel_height = int(target.get("pixel_height", 0))
    if pixel_height <= 0:
        return None
    return (cfg.focal_length * cfg.real_object_height) / float(pixel_height)


class ColorTargetDetector:
    """HSV blob detector driven by the detection block of bruno_config.json."""

    def __init__(self, cfg: PickPlaceConfig, json_cfg: dict):
        detection = (json_cfg or {}).get("detection", {}) or {}
        self.cfg = cfg
        self.ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, bounds in (detection.get("color_detection", {}) or {}).items():
            # config keys are e.g. "blue_plastic" -> expose as "blue"
            key = name.split("_")[0].lower()
            try:
                lower = np.array(bounds["lower_hsv"], dtype=np.uint8)
                upper = np.array(bounds["upper_hsv"], dtype=np.uint8)
            except Exception:
                continue
            self.ranges[key] = (lower, upper)
        morph = detection.get("morphology", {}) or {}
        self.kernel_size = int(morph.get("kernel_size", 3))
        self.iterations = int(morph.get("iterations", 2))
        self.reject_reason: Optional[str] = None

    def available_colors(self) -> Tuple[str, ...]:
        return tuple(self.ranges.keys())

    def detect(self, frame, color: str) -> Optional[Dict[str, Any]]:
        """Largest in-range blob passing the size filter, or None."""
        candidates = list(self.ranges.items())
        if color != "any":
            if color not in self.ranges:
                LOG.warning("No HSV range for color '%s'; falling back to any", color)
            else:
                candidates = [(color, self.ranges[color])]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        best: Optional[Dict[str, Any]] = None
        rejects: Dict[str, int] = {}

        for name, (lower, upper) in candidates:
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.cfg.min_area:
                    rejects["min_area"] = rejects.get("min_area", 0) + 1
                    continue
                if area > self.cfg.max_area:
                    rejects["max_area"] = rejects.get("max_area", 0) + 1
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if h <= 0 or w <= 0:
                    continue
                aspect = h / float(w)
                if not (self.cfg.min_aspect_ratio <= aspect <= self.cfg.max_aspect_ratio):
                    rejects["aspect"] = rejects.get("aspect", 0) + 1
                    continue
                if best is None or area > best["area"]:
                    best = {
                        "color": name,
                        "area": area,
                        "aspect": aspect,
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "center": (int(x + w / 2), int(y + h / 2)),
                        "pixel_height": int(h),
                        "pixel_width": int(w),
                        # A bottle close enough to grasp runs off the top or
                        # bottom of the frame; its pixel height then understates
                        # the distance badly, so flag it for distance_cm().
                        "clipped": bool(y <= 1 or (y + h) >= frame.shape[0] - 1),
                    }

        # Search failing tells you nothing on its own — say which filter ate the
        # blobs, so tune_detection.py is only needed when this is not enough.
        self.reject_reason = (
            ", ".join(f"{k}x{v}" for k, v in sorted(rejects.items())) if best is None and rejects else None
        )
        if best is None and self.reject_reason:
            LOG.debug("no target: blobs rejected by %s", self.reject_reason)
        return best

    def distance_cm(self, target: Dict[str, Any]) -> Optional[float]:
        return estimate_distance_cm(self.cfg, target)


class YoloTargetDetector:
    """Class-based detector: finds the object by *what it is*, not its color.

    HSV cannot see a clear PET bottle — it shows whatever is behind it, so the
    "clear" range matches the wall as readily as the target. YOLO boxes the
    whole bottle regardless of transparency, which also keeps the pinhole
    distance estimate honest: the box height really is the bottle's height.

    When the plan named a color and several bottles are visible, the requested
    HSV range breaks the tie by which box contains the most in-range pixels.
    """

    def __init__(self, cfg: PickPlaceConfig, color_detector: ColorTargetDetector):
        self.cfg = cfg
        self.colors = color_detector
        self.model = None
        self.reject_reason: Optional[str] = None
        self.available = False
        if not cfg.use_yolo:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            LOG.warning("YOLO requested but ultralytics is not installed (%s); using HSV only", exc)
            return
        try:
            self.model = YOLO(cfg.yolo_model)
            self.available = True
            LOG.info(
                "YOLO enabled model=%s conf=%.2f classes=%s",
                cfg.yolo_model,
                cfg.yolo_conf,
                ",".join(cfg.yolo_classes),
            )
        except Exception as exc:
            LOG.warning("YOLO init failed (%s); using HSV only", exc)

    def _wanted_class(self, name: str) -> bool:
        return name.strip().lower() in self.cfg.yolo_classes

    def _color_score(self, frame, bbox: Tuple[int, int, int, int], color: str) -> float:
        """Fraction of the box matching the requested HSV range, 0.0 if unknown."""
        bounds = self.colors.ranges.get(color)
        if bounds is None:
            return 0.0
        x, y, w, h = bbox
        patch = frame[y : y + h, x : x + w]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, bounds[0], bounds[1])
        return float(np.count_nonzero(mask)) / float(patch.shape[0] * patch.shape[1])

    def detect(self, frame, color: str) -> Optional[Dict[str, Any]]:
        self.reject_reason = None
        if not self.available or self.model is None:
            return None
        try:
            results = self.model.predict(
                source=frame,
                verbose=False,
                conf=max(0.01, min(0.99, self.cfg.yolo_conf)),
                imgsz=self.cfg.yolo_imgsz,
                max_det=10,
            )
        except Exception as exc:
            LOG.debug("YOLO inference failed: %s", exc)
            return None
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        names = getattr(results[0], "names", {}) or {}

        frame_h, frame_w = frame.shape[:2]
        candidates = []
        off_class = 0
        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = [int(max(0, v)) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i].item())
                label = str(names.get(int(boxes.cls[i].item()), "")).lower()
            except Exception:
                continue
            if not self._wanted_class(label):
                off_class += 1
                continue
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0 or (w * h) < self.cfg.yolo_min_area:
                continue
            bbox = (x1, y1, w, h)
            candidates.append(
                {
                    "color": color,
                    "label": label,
                    "conf": conf,
                    "area": float(w * h),
                    "aspect": h / float(w),
                    "bbox": bbox,
                    "center": (int(x1 + w / 2), int(y1 + h / 2)),
                    "pixel_height": int(h),
                    "pixel_width": int(w),
                    "clipped": bool(y1 <= 1 or y2 >= frame_h - 1),
                    "color_score": self._color_score(frame, bbox, color) if color != "any" else 0.0,
                }
            )

        if not candidates:
            if off_class:
                self.reject_reason = f"yolo saw {off_class} object(s), none of class {'/'.join(self.cfg.yolo_classes)}"
            return None
        # A color match is only a tie-breaker: a confident bottle detection that
        # fails the HSV check is still a bottle, and for clear plastic it always
        # will fail. Only prefer color when it is decisive.
        colored = [c for c in candidates if c["color_score"] >= 0.15]
        pool = colored if colored else candidates
        best = max(pool, key=lambda c: c["area"])
        if len(candidates) > 1:
            LOG.debug(
                "yolo: %d bottles, picked area=%.0f conf=%.2f color_score=%.2f",
                len(candidates),
                best["area"],
                best["conf"],
                best["color_score"],
            )
        return best

    def distance_cm(self, target: Dict[str, Any]) -> Optional[float]:
        return estimate_distance_cm(self.cfg, target)


class TargetDetector:
    """YOLO first, HSV as the fallback — one interface for the state machine."""

    def __init__(self, cfg: PickPlaceConfig, json_cfg: dict):
        self.cfg = cfg
        self.colors = ColorTargetDetector(cfg, json_cfg)
        self.yolo = YoloTargetDetector(cfg, self.colors)
        self.reject_reason: Optional[str] = None
        self.last_source = "hsv"

    @property
    def yolo_active(self) -> bool:
        return self.yolo.available

    def detect(self, frame, color: str) -> Optional[Dict[str, Any]]:
        if self.yolo.available:
            target = self.yolo.detect(frame, color)
            if target is not None:
                self.last_source = "yolo"
                self.reject_reason = None
                return target
            # Falling back keeps a tape-wrapped or brightly colored bottle
            # working on the frames where the class detector misses.
            fallback = self.colors.detect(frame, color)
            if fallback is not None:
                self.last_source = "hsv"
                self.reject_reason = None
                return fallback
            self.reject_reason = self.yolo.reject_reason or self.colors.reject_reason
            return None
        target = self.colors.detect(frame, color)
        self.last_source = "hsv"
        self.reject_reason = None if target is not None else self.colors.reject_reason
        return target

    def distance_cm(self, target: Dict[str, Any]) -> Optional[float]:
        return estimate_distance_cm(self.cfg, target)


def _raw_content(data: Dict[str, Any]) -> str:
    """The model's message text, for logging when a response is unusable."""
    try:
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return "<unreadable>"
    text = (text or "").strip().replace("\n", " ")
    return text[:300] if text else "<empty>"


class PickPlaceVLM:
    """Single-purpose VLM client for planning and grasp verification."""

    def __init__(self, cfg: PickPlaceConfig):
        self.cfg = cfg

    def _call(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return post_chat_completion(
                api_base=self.cfg.vlm_base,
                api_key=self.cfg.vlm_api_key,
                payload=payload,
                timeout_sec=max(0.5, self.cfg.vlm_timeout_ms / 1000.0),
            )
        except Exception as exc:
            LOG.error("VLM request failed: %s", exc)
            return None

    def plan(self, frame, instruction: str) -> Optional[Dict[str, Any]]:
        try:
            payload = build_plan_payload(frame, instruction, self.cfg.vlm_model)
        except Exception as exc:
            LOG.error("Plan encode failed: %s", exc)
            return None
        data = self._call(payload)
        if data is None:
            return None
        plan, reason = parse_plan_response(data)
        if plan is None:
            LOG.error("Plan parse failed: %s — model said: %s", reason, _raw_content(data))
            return None
        if plan["confidence"] < self.cfg.vlm_min_confidence:
            # Small VLMs tend to echo the schema back rather than describe the
            # scene, so show what was actually returned.
            LOG.error(
                "Plan confidence too low: %.2f (%s) — model said: %s",
                plan["confidence"],
                plan["target_description"],
                _raw_content(data),
            )
            return None
        LOG.debug("plan raw: %s", _raw_content(data))
        return plan

    def check_grasp(self, frame, target_description: str) -> Optional[bool]:
        """True/False if the model answered, None if the check was unavailable."""
        try:
            payload = build_grasp_check_payload(frame, target_description, self.cfg.vlm_model)
        except Exception as exc:
            LOG.error("Grasp check encode failed: %s", exc)
            return None
        data = self._call(payload)
        if data is None:
            return None
        result, reason = parse_grasp_check_response(data)
        if result is None:
            LOG.warning("Grasp check parse failed: %s", reason)
            return None
        if result["confidence"] < self.cfg.vlm_min_confidence:
            return None
        return result["holding"]


class PickPlaceRunner:
    """State machine tying perception, base motion, and the arm together."""

    def __init__(self, cfg: PickPlaceConfig, instruction: str, dry_run: bool = False):
        self.cfg = cfg
        self.instruction = instruction
        self.dry_run = dry_run

        json_cfg = load_json_config()
        self.detector = TargetDetector(cfg, json_cfg)
        self.vlm = PickPlaceVLM(cfg)
        self.arm = ArmController(cfg=ArmConfig.from_dict(json_cfg), dry_run=dry_run)
        self.arm_cfg = self.arm.cfg

        self.camera = make_camera(cfg.mode, retry_attempts=3, retry_delay=2.0)
        self.last_frame = None

        self.motion = None
        if MecanumWrapper is not None and not dry_run:
            try:
                self.motion = MecanumWrapper(forward_speed=cfg.forward_speed, turn_speed=cfg.forward_speed)
            except Exception as exc:
                LOG.warning("Mecanum unavailable: %s", exc)

        self.ultra = None
        if UltrasonicRGB is not None and not dry_run:
            try:
                self.ultra = UltrasonicRGB()
            except Exception as exc:
                LOG.warning("Ultrasonic unavailable: %s", exc)

        self.state = State.PLAN
        self.plan: Optional[Dict[str, Any]] = None
        self.state_entered = time.time()
        self.grasp_attempts = 0
        self.grasp_pose: Tuple[float, float, float] = (
            self.arm_cfg.home_position[0],
            0.0,
            self.arm_cfg.pickup_height,
        )

    # ---------- helpers ----------

    def set_state(self, state: State) -> None:
        if state is not self.state:
            LOG.info("state: %s -> %s", self.state.value, state.value)
        self.state = state
        self.state_entered = time.time()

    def in_state_for(self) -> float:
        return time.time() - self.state_entered

    def grab_frame(self):
        frame = read_or_reconnect(self.camera, self.last_frame)
        if frame is not None:
            self.last_frame = frame
        return frame

    def stop_base(self) -> None:
        if self.motion is not None:
            self.motion.stop()

    def base_blocked(self) -> bool:
        """Ultrasonic veto — the target itself sits beyond the grasp standoff."""
        if self.ultra is None:
            return False
        try:
            d = self.ultra.get_distance_cm()
        except Exception:
            return False
        if d is not None and d <= self.cfg.ultra_danger_cm:
            LOG.warning("ultrasonic block at %.1f cm", d)
            self.stop_base()
            return True
        return False

    def nudge(self, rotation: float = 0.0, speed: Optional[int] = None, duration: Optional[float] = None) -> None:
        """Short open-loop base motion; the loop re-perceives after each nudge."""
        if self.motion is None:
            LOG.info("[dry] nudge speed=%s rot=%.2f", speed, rotation)
            time.sleep(0.05)
            return
        t = float(duration if duration is not None else self.cfg.nudge_s)
        v = int(speed if speed is not None else 0)
        try:
            self.motion.set_velocity(v, 90, rotation)
            time.sleep(t)
        finally:
            self.motion.stop()

    def target_pose(self, offset_x: float, distance_cm: float) -> Tuple[float, float, float]:
        """Map a centered target at the standoff to the calibrated grasp pose.

        The base drives until the object is centered at the standoff distance, so
        the grasp pose is fixed by calibration; the residual horizontal offset is
        applied as a small y correction within the arm's workspace.
        """
        grasp = getattr(self.arm_cfg, "grasp_position", None) or (
            self.arm_cfg.home_position[0],
            0.0,
            self.arm_cfg.pickup_height,
        )
        # offset_x is normalized [-1, 1]; scale to a conservative +/- 4 cm.
        y_correction = max(-4.0, min(4.0, -offset_x * 4.0))
        return (float(grasp[0]), float(grasp[1]) + y_correction, float(grasp[2]))

    # ---------- states ----------

    def do_plan(self, frame) -> None:
        if self.cfg.plan_override is not None:
            plan = dict(self.cfg.plan_override)
            LOG.info("planning: skipped, using --target-color/--place override")
        else:
            LOG.info("planning: %r", self.instruction)
            plan = self.vlm.plan(frame, self.instruction)
        if plan is None:
            LOG.error("no usable plan — aborting")
            self.set_state(State.FAILED)
            return
        self.plan = plan
        LOG.info(
            "plan: pick '%s' (color=%s) -> place %s (conf %.2f)",
            plan["target_description"],
            plan["target_color"],
            plan["place"],
            plan["confidence"],
        )
        self.arm.home()
        self.set_state(State.SEARCH)

    def do_search(self, frame) -> None:
        if self.in_state_for() > self.cfg.search_timeout_s:
            LOG.error("target not found within %.0fs", self.cfg.search_timeout_s)
            self.set_state(State.FAILED)
            return
        target = self.detector.detect(frame, self.plan["target_color"])
        if target is not None:
            LOG.info(
                "target acquired via %s: %s area=%.0f aspect=%.1f",
                self.detector.last_source,
                target.get("label", target["color"]),
                target["area"],
                target["aspect"],
            )
            self.set_state(State.APPROACH)
            return
        if self.detector.reject_reason:
            LOG.info("candidates rejected by %s", self.detector.reject_reason)
        self.nudge(rotation=self.cfg.turn_rotation, duration=self.cfg.search_turn_s)

    def do_approach(self, frame) -> None:
        if self.in_state_for() > self.cfg.approach_timeout_s:
            LOG.error("approach timed out")
            self.set_state(State.FAILED)
            return
        target = self.detector.detect(frame, self.plan["target_color"])
        if target is None:
            LOG.info("target lost during approach")
            self.stop_base()
            self.set_state(State.SEARCH)
            return

        width = frame.shape[1]
        offset_x = (target["center"][0] - width / 2.0) / (width / 2.0)
        distance = self.detector.distance_cm(target)

        if abs(offset_x) > self.cfg.center_deadband:
            rotation = self.cfg.turn_rotation if offset_x > 0 else -self.cfg.turn_rotation
            self.nudge(rotation=rotation)
            return

        if distance is None:
            LOG.warning("no distance estimate; nudging forward")
            self.nudge(speed=self.cfg.forward_speed)
            return

        LOG.info("target centered at %.1f cm (offset %.2f)", distance, offset_x)
        if distance <= self.cfg.grasp_standoff_cm + self.cfg.standoff_tolerance_cm:
            self.stop_base()
            self.set_state(State.ALIGN)
            return
        if self.base_blocked():
            LOG.error("blocked before reaching standoff")
            self.set_state(State.FAILED)
            return
        self.nudge(speed=self.cfg.forward_speed)

    def do_align(self, frame) -> None:
        """Final re-check at standoff, then commit to the grasp pose."""
        target = self.detector.detect(frame, self.plan["target_color"])
        if target is None:
            LOG.info("target lost at standoff")
            self.set_state(State.SEARCH)
            return
        width = frame.shape[1]
        offset_x = (target["center"][0] - width / 2.0) / (width / 2.0)
        distance = self.detector.distance_cm(target) or self.cfg.grasp_standoff_cm
        self.grasp_pose = self.target_pose(offset_x, distance)
        LOG.info("grasp pose (%.1f, %.1f, %.1f)", *self.grasp_pose)
        self.set_state(State.GRASP)

    def do_grasp(self, frame) -> None:
        self.grasp_attempts += 1
        if self.arm.pick(self.grasp_pose):
            self.set_state(State.VERIFY)
            return
        LOG.warning("grasp attempt %d failed (unreachable)", self.grasp_attempts)
        self.arm.home()
        if self.grasp_attempts > self.cfg.grasp_retries:
            self.set_state(State.FAILED)
        else:
            self.set_state(State.APPROACH)

    def do_verify(self, frame) -> None:
        holding = self.vlm.check_grasp(frame, self.plan["target_description"])
        if holding is False and self.grasp_attempts <= self.cfg.grasp_retries:
            LOG.warning("gripper appears empty — retrying")
            self.arm.home()
            self.set_state(State.APPROACH)
            return
        if holding is None:
            LOG.info("grasp check inconclusive — proceeding")
        self.set_state(State.PLACE)

    def do_place(self, frame) -> None:
        place = self.plan["place"]
        if place in ("left", "right"):
            rotation = -self.cfg.turn_rotation if place == "left" else self.cfg.turn_rotation
            self.nudge(rotation=rotation, duration=0.8)
        elif place == "ahead":
            self.nudge(speed=self.cfg.forward_speed, duration=0.5)
        if self.arm.place():
            LOG.info("placed at %s", place)
            self.arm.home()
            self.set_state(State.DONE)
        else:
            LOG.error("drop pose unreachable")
            self.set_state(State.FAILED)

    # ---------- loop ----------

    def run(self) -> int:
        handlers = {
            State.PLAN: self.do_plan,
            State.SEARCH: self.do_search,
            State.APPROACH: self.do_approach,
            State.ALIGN: self.do_align,
            State.GRASP: self.do_grasp,
            State.VERIFY: self.do_verify,
            State.PLACE: self.do_place,
        }
        try:
            if not self.camera.open():
                LOG.error("camera failed to open")
                return 2
            while self.state not in (State.DONE, State.FAILED):
                frame = self.grab_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                handlers[self.state](frame)
        except KeyboardInterrupt:
            LOG.info("interrupted")
        finally:
            self.stop_base()
            try:
                self.arm.relax()
            except Exception:
                pass
            try:
                self.camera.release()
            except Exception:
                pass
        if self.state is State.DONE:
            LOG.info("✅ pick and place complete")
            return 0
        LOG.error("❌ pick and place failed in state %s", self.state.value)
        return 1


def build_config(args) -> PickPlaceConfig:
    json_cfg = load_json_config()
    distance = (json_cfg or {}).get("distance_estimation", {}) or {}
    size_filter = ((json_cfg or {}).get("detection", {}) or {}).get("size_filter", {}) or {}
    return PickPlaceConfig(
        mode=args.mode,
        min_area=int(size_filter.get("min_area", 500)),
        max_area=int(size_filter.get("max_area", 150000)),
        min_aspect_ratio=float(size_filter.get("min_aspect_ratio", 1.2)),
        max_aspect_ratio=float(size_filter.get("max_aspect_ratio", 4.5)),
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_classes=tuple(c.strip().lower() for c in args.yolo_classes.split(",") if c.strip()),
        yolo_imgsz=args.yolo_imgsz,
        focal_length=float(distance.get("focal_length", 500)),
        real_object_height=float(distance.get("real_bottle_height", 20)),
        real_object_width=float(distance.get("real_bottle_diameter", 7)),
        grasp_standoff_cm=args.standoff,
        forward_speed=args.speed,
        vlm_base=args.vlm_base,
        vlm_model=args.vlm_model,
        vlm_api_key=get_env_str("BRUNO_VLM_API_KEY", "lm-studio"),
        vlm_timeout_ms=args.vlm_timeout_ms,
        vlm_min_confidence=args.vlm_min_confidence,
        plan_override=(
            {
                "target_description": f"{args.target_color} object",
                "target_color": args.target_color,
                "place": args.place,
                "confidence": 1.0,
            }
            if args.target_color
            else None
        ),
    )


def main() -> int:
    load_env(os.path.join(REPO_ROOT, ".env"))
    parser = argparse.ArgumentParser(description="Prompt-driven pick and place")
    parser.add_argument(
        "--prompt", default="", help="natural-language instruction (not needed with --target-color)"
    )
    parser.add_argument("--mode", default=get_env_str("CAM_MODE", "builtin"), choices=["builtin", "external"])
    parser.add_argument("--standoff", type=float, default=get_env_float("BRUNO_PP_STANDOFF_CM", 18.0))
    parser.add_argument("--speed", type=int, default=get_env_int("BRUNO_PP_SPEED", 25))
    parser.add_argument("--vlm-base", default=get_env_str("BRUNO_VLM_LOCAL_BASE", "http://127.0.0.1:8081/v1"))
    parser.add_argument("--vlm-model", default=get_env_str("BRUNO_VLM_LOCAL_MODEL", "gemma3"))
    parser.add_argument("--vlm-timeout-ms", type=int, default=get_env_int("BRUNO_PP_VLM_TIMEOUT_MS", 8000))
    parser.add_argument("--vlm-min-confidence", type=float, default=get_env_float("BRUNO_PP_VLM_MIN_CONF", 0.35))
    parser.add_argument(
        "--target-color",
        choices=sorted(ALLOWED_COLORS),
        default=None,
        help="skip the VLM and track this color directly (pairs with --place)",
    )
    parser.add_argument(
        "--place",
        choices=sorted(ALLOWED_PLACEMENTS),
        default="home",
        help="where to drop the object when --target-color is used",
    )
    parser.add_argument(
        "--no-yolo",
        dest="use_yolo",
        action="store_false",
        default=get_env_int("BRUNO_PP_USE_YOLO", 1) == 1,
        help="track by HSV color only (a clear bottle will not be found)",
    )
    parser.add_argument("--yolo-model", default=get_env_str("BRUNO_YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--yolo-conf", type=float, default=get_env_float("BRUNO_PP_YOLO_CONF", 0.35))
    parser.add_argument(
        "--yolo-classes",
        default=get_env_str("BRUNO_PP_YOLO_CLASSES", "bottle"),
        help="comma-separated COCO class names to accept",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=get_env_int("BRUNO_PP_YOLO_IMGSZ", 320))
    parser.add_argument("--dry-run", action="store_true", help="no motion; perception and planning only")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.prompt and not args.target_color:
        parser.error("give --prompt, or --target-color to skip the VLM")

    if args.debug:
        LOG.setLevel(logging.DEBUG)

    cfg = build_config(args)
    runner = PickPlaceRunner(cfg, args.prompt, dry_run=args.dry_run)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
