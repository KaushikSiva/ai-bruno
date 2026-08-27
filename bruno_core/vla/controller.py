"""Arming, bounds, idempotency, timed stop, and the dead-man switch.

Sits between whatever is issuing commands and the driver, and is deliberately
paranoid: it starts disarmed, clamps every speed and duration, ignores repeated
request ids, stops the chassis when a command's time is up, and disarms if
commands stop arriving. A driver error disarms too, on the principle that a
robot which just failed a motion command should not be handed the next one.

One limit is not enforceable here and should not be mistaken for one: arm and
gripper commands are *position targets*. Once issued, a servo will travel to
them whatever the dead-man timer does afterwards. Keep arm durations short,
keep the workspace clear, and keep a physical emergency stop within reach.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, Optional, Set

from .calibration import CalibrationProfile
from .contracts import POSITION_ACTIONS, ROTATION_ACTION, Action
from .drivers.base import MotionDriver

LOG = logging.getLogger("bruno_core.vla.controller")

SEEN_REQUEST_LIMIT = 256


class RobotNotArmed(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def limits_from_env() -> Dict[str, Any]:
    """Controller bounds from the environment.

    Shared so that a command means the same thing whether it goes through the
    bridge or straight into a driver. They used to diverge: the bridge read
    these variables and teleop did not, so raising a limit for one silently
    left the other at its default.
    """
    return {
        "watchdog_seconds": _env_float("BRUNO_COMMAND_WATCHDOG_SECONDS", 2.5),
        "max_speed": _env_float("BRUNO_BRIDGE_MAX_SPEED", 0.5),
        "max_duration_ms": _env_int("BRUNO_BRIDGE_MAX_DURATION_MS", 500),
        "min_confidence": _env_float("BRUNO_BRIDGE_MIN_CONFIDENCE", 0.5),
        "max_rotation_deg": _env_float("BRUNO_MAX_ROTATION_DEGREES", 90.0),
        "max_rotation_duration_ms": _env_int("BRUNO_MAX_ROTATION_DURATION_MS", 4000),
    }


class RobotController:
    def __init__(
        self,
        driver: MotionDriver,
        profile: Optional[CalibrationProfile] = None,
        watchdog_seconds: float = 2.5,
        max_speed: float = 0.5,
        max_duration_ms: int = 500,
        min_confidence: float = 0.5,
        max_rotation_deg: float = 90.0,
        max_rotation_duration_ms: int = 4000,
    ):
        self.driver = driver
        self.profile = profile or getattr(driver, "profile", None) or CalibrationProfile()
        self.watchdog_seconds = max(0.2, watchdog_seconds)
        self.max_speed = max(0.0, min(1.0, max_speed))
        self.max_duration_ms = max(1, max_duration_ms)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.max_rotation_deg = max(1.0, min(180.0, max_rotation_deg))
        self.max_rotation_duration_ms = max(1, max_rotation_duration_ms)

        self._lock = threading.RLock()
        self._armed = False
        self._last_command_at = time.monotonic()
        self._last_action = Action.stop("controller initialized")
        self._seen_order: Deque[str] = deque()
        self._seen: Set[str] = set()
        self._stop_timer: Optional[threading.Timer] = None
        self._closed = threading.Event()
        self._watchdog_trips = 0
        self._last_driver_error = ""

        if not self._stop_locked("controller initialized"):
            raise RuntimeError("the driver did not accept the startup stop command")
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="bruno-vla-deadman", daemon=True
        )
        self._watchdog.start()

    # ---------- public API ----------

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        with self._lock:
            stopped = self._stop_locked("arm state changed")
            if armed and not stopped:
                self._armed = False
                raise RuntimeError("cannot arm because the stop command failed")
            self._armed = bool(armed)
            self._last_command_at = time.monotonic()
            LOG.info("Robot %s", "ARMED" if self._armed else "disarmed")
            return self.status()

    def execute(self, action: Action) -> Dict[str, Any]:
        with self._lock:
            if action.request_id in self._seen:
                result = self.status()
                result["duplicate"] = True
                return result
            if not self._armed and action.action != "stop":
                raise RobotNotArmed("robot is disarmed")

            bounded = self._bound(action)
            self._remember(bounded.request_id)
            self._last_command_at = time.monotonic()
            self._cancel_timer_locked()
            try:
                self.driver.apply(bounded)
            except Exception as exc:
                self._armed = False
                # Record the failure after the stop, which clears the field on
                # its own success -- otherwise the cause is lost immediately.
                self._stop_locked("driver error; robot disarmed")
                self._last_driver_error = f"{type(exc).__name__}: {exc}"
                raise
            self._last_action = bounded
            self._last_driver_error = ""
            # Position targets finish on their own. Only velocity commands need
            # a timer to end them, and only those can be cancelled by one.
            if bounded.action != "stop" and bounded.action not in POSITION_ACTIONS:
                timer = threading.Timer(bounded.duration_ms / 1000.0, self._duration_expired)
                timer.daemon = True
                self._stop_timer = timer
                timer.start()
            result = self.status()
            result["duplicate"] = False
            result["executed"] = bounded.to_dict()
            return result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "armed": self._armed,
                "last_command_age_ms": round((time.monotonic() - self._last_command_at) * 1000),
                "last_action": self._last_action.to_dict(),
                "watchdog_seconds": self.watchdog_seconds,
                "watchdog_trips": self._watchdog_trips,
                "last_driver_error": self._last_driver_error,
                "driver": self.driver.status(),
            }

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self._armed = False
            self._stop_locked("controller closed")
        self._watchdog.join(timeout=1.0)
        self.driver.close()

    # ---------- internals ----------

    def _bound(self, action: Action) -> Action:
        if action.action != "stop" and action.confidence < self.min_confidence:
            return Action.stop(
                f"confidence {action.confidence:.2f} below {self.min_confidence:.2f}",
                request_id=action.request_id,
                confidence=action.confidence,
            )
        bounded = replace(
            action,
            speed=min(action.speed, self.max_speed),
            duration_ms=min(action.duration_ms, self.max_duration_ms)
            if action.action not in POSITION_ACTIONS
            else action.duration_ms,
        )
        if bounded.action == ROTATION_ACTION:
            angle = max(-self.max_rotation_deg, min(self.max_rotation_deg, bounded.angle_deg))
            # Duration comes from the shared calibration, so the same rotation
            # request spins for the same wall-clock time in sim and on hardware.
            duration = self.profile.chassis.rotation_duration_ms(angle, bounded.speed)
            bounded = replace(
                bounded,
                angle_deg=angle,
                duration_ms=max(1, min(self.max_rotation_duration_ms, duration)),
            )
        return bounded

    def _duration_expired(self) -> None:
        with self._lock:
            self._stop_timer = None
            if not self._stop_locked("command duration expired"):
                self._armed = False

    def _watchdog_loop(self) -> None:
        interval = min(0.1, self.watchdog_seconds / 4.0)
        while not self._closed.wait(interval):
            with self._lock:
                idle = time.monotonic() - self._last_command_at
                if self._armed and idle > self.watchdog_seconds:
                    self._armed = False
                    self._watchdog_trips += 1
                    LOG.warning("Dead-man timeout after %.1fs; robot disarmed", idle)
                    self._stop_locked("dead-man timeout; robot disarmed")

    def _stop_locked(self, reason: str) -> bool:
        self._cancel_timer_locked()
        try:
            self.driver.stop()
        except Exception as exc:
            self._last_driver_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("Stop command failed while handling: %s", reason)
            self._last_action = Action.stop(f"stop command failed: {reason}")
            return False
        self._last_driver_error = ""
        self._last_action = Action.stop(reason)
        return True

    def _cancel_timer_locked(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def _remember(self, request_id: str) -> None:
        self._seen.add(request_id)
        self._seen_order.append(request_id)
        while len(self._seen_order) > SEEN_REQUEST_LIMIT:
            self._seen.discard(self._seen_order.popleft())
