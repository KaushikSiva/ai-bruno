"""The bounded action vocabulary shared by the teleop CLI, sim, and hardware.

Schema v3 extends the ``bruno.vla.action.v2`` set from the bruno_vla project
with the arm commands this repo needs:

- ``rotate_arm_by_deg`` turns one named joint by a signed angle. This is the
  command Cartesian jogging cannot express -- ``move_arm_left`` slides the tool
  sideways, whereas rotating the base joint swings the whole arm about its
  column. Joint space is also the only place sim and hardware can be compared
  directly, since a Cartesian pose hides which elbow branch produced it.
- ``move_arm_forward`` / ``move_arm_backward`` complete the Cartesian jog set.
- ``home_arm`` returns to the calibrated home pose, which teleop needs as a
  known-good recovery.

One deliberate break from v2: rotation is **positive counter-clockwise**, so a
positive ``angle_deg`` turns the chassis to its left and swings the arm's base
joint to its left. v2 defined chassis rotation the other way round, which put
it at odds with the right-hand rule the MuJoCo model and every arm joint use --
``rotate_by_deg 30`` and ``rotate_arm_by_deg base 30`` would have moved in
opposite directions. v3 is what this repo speaks; nothing here parses v2.

Nothing here executes anything. A payload that parses is merely well formed;
the controller still applies speed, duration, arming, and confidence limits,
and the driver still applies workspace and servo limits.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .kinematics import JOINT_NAMES

ACTION_SCHEMA = "bruno.vla.action.v3"

CHASSIS_TRANSLATION_ACTIONS = frozenset({"up", "down", "left", "right"})
ROTATION_ACTION = "rotate_by_deg"
CHASSIS_ACTIONS = frozenset({ROTATION_ACTION} | CHASSIS_TRANSLATION_ACTIONS)

ARM_CARTESIAN_ACTIONS = frozenset(
    {
        "move_arm_up",
        "move_arm_down",
        "move_arm_left",
        "move_arm_right",
        "move_arm_forward",
        "move_arm_backward",
    }
)
ARM_ROTATION_ACTION = "rotate_arm_by_deg"
HOME_ARM_ACTION = "home_arm"
ARM_ACTIONS = frozenset({ARM_ROTATION_ACTION, HOME_ARM_ACTION} | ARM_CARTESIAN_ACTIONS)

GRIPPER_ACTIONS = frozenset({"open_gripper", "close_gripper"})

# Actions whose motion is a position target rather than a timed velocity. These
# cannot be recalled once issued, which is why the dead-man timer is not a
# substitute for a physical emergency stop during manipulator work.
POSITION_ACTIONS = frozenset(ARM_ACTIONS | GRIPPER_ACTIONS)

ALLOWED_ACTIONS = frozenset({"stop"} | CHASSIS_ACTIONS | ARM_ACTIONS | GRIPPER_ACTIONS)

MAX_DURATION_MS = 10_000


class ContractError(ValueError):
    """Raised when a payload does not match the Bruno action contract."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a number") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ContractError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Action:
    action: str
    speed: float = 0.0
    duration_ms: int = 0
    confidence: float = 1.0
    reason: str = "no reason"
    angle_deg: float = 0.0
    joint: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema: str = ACTION_SCHEMA
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def stop(cls, reason: str, request_id: str = "", confidence: float = 1.0) -> "Action":
        return cls(
            action="stop",
            speed=0.0,
            duration_ms=0,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason[:300],
            request_id=request_id or str(uuid.uuid4()),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], request_id: str = "") -> "Action":
        if not isinstance(data, Mapping):
            raise ContractError("action must be a JSON object")
        schema = str(data.get("schema", ACTION_SCHEMA))
        if schema != ACTION_SCHEMA:
            raise ContractError(f"unsupported action schema: {schema}")

        action = str(data.get("action", "")).strip().lower()
        if action not in ALLOWED_ACTIONS:
            raise ContractError(f"unsupported action: {action}")

        speed = _finite_float(data.get("speed", 0.0), "speed")
        angle_deg = _finite_float(data.get("angle_deg", 0.0), "angle_deg")
        confidence = _finite_float(data.get("confidence", 1.0), "confidence")
        try:
            duration_ms = int(data.get("duration_ms", 0))
        except (TypeError, ValueError) as exc:
            raise ContractError("duration_ms must be an integer") from exc
        joint = str(data.get("joint", "")).strip().lower()

        if not 0.0 <= speed <= 1.0:
            raise ContractError("speed must be between 0 and 1")
        if not 0.0 <= confidence <= 1.0:
            raise ContractError("confidence must be between 0 and 1")
        if not -180.0 <= angle_deg <= 180.0:
            raise ContractError("angle_deg must be between -180 and 180")
        if not 0 <= duration_ms <= MAX_DURATION_MS:
            raise ContractError(f"duration_ms must be between 0 and {MAX_DURATION_MS}")

        reason = str(data.get("reason", "no reason")).strip()[:300] or "no reason"
        result_id = str(data.get("request_id") or request_id or uuid.uuid4())[:128]

        if action == "stop":
            speed, duration_ms, angle_deg, joint = 0.0, 0, 0.0, ""
        elif action == ROTATION_ACTION:
            joint = ""
            if speed <= 0.0:
                raise ContractError("rotate_by_deg requires a positive speed")
            if abs(angle_deg) < 1.0:
                raise ContractError("rotate_by_deg requires abs(angle_deg) of at least 1")
        elif action == ARM_ROTATION_ACTION:
            if joint not in JOINT_NAMES:
                raise ContractError(
                    f"rotate_arm_by_deg requires joint in {', '.join(JOINT_NAMES)}"
                )
            if abs(angle_deg) < 0.1:
                raise ContractError("rotate_arm_by_deg requires abs(angle_deg) of at least 0.1")
            speed = 0.0
        elif action in GRIPPER_ACTIONS or action == HOME_ARM_ACTION:
            speed, angle_deg, joint = 0.0, 0.0, ""
        elif action in ARM_CARTESIAN_ACTIONS:
            angle_deg, joint = 0.0, ""
            if speed <= 0.0:
                raise ContractError("arm jog actions require a positive speed")
        else:  # chassis translation
            angle_deg, joint = 0.0, ""
            if speed <= 0.0 or duration_ms <= 0:
                raise ContractError("movement actions require positive speed and duration_ms")

        return cls(
            action=action,
            speed=speed,
            duration_ms=duration_ms,
            confidence=confidence,
            reason=reason,
            angle_deg=angle_deg,
            joint=joint,
            request_id=result_id,
            schema=schema,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "timestamp_ms": self.timestamp_ms,
            "action": self.action,
            "speed": round(self.speed, 4),
            "duration_ms": self.duration_ms,
            "angle_deg": round(self.angle_deg, 3),
            "joint": self.joint,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }

    def describe(self) -> str:
        if self.action == ROTATION_ACTION:
            return f"{self.action} {self.angle_deg:+.1f}deg @ {self.speed:.2f}"
        if self.action == ARM_ROTATION_ACTION:
            return f"{self.action} {self.joint} {self.angle_deg:+.1f}deg"
        if self.action in POSITION_ACTIONS:
            return f"{self.action} ({self.duration_ms}ms)"
        if self.action == "stop":
            return "stop"
        return f"{self.action} @ {self.speed:.2f} for {self.duration_ms}ms"
