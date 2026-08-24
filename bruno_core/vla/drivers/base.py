"""Driver protocol, shared action dispatch, and the no-hardware mock.

``BaseDriver`` holds every decision a command implies -- which joints to target,
which workspace clamp applied, how long a rotation should take -- and leaves
subclasses only the final step of pushing physical units at an actuator. Both
the simulator and the robot inherit it, so there is exactly one implementation
of what an action *means* and two of how it is *delivered*.

Subclasses implement three hooks, all in physical units:

``_drive(vx_cmps, vy_cmps, wz_deg_s)``   chassis velocity, robot frame
``_set_joints(joints_deg, duration_ms)`` the four arm joints, model convention
``_set_gripper(opened, duration_ms)``    jaws open or closed

Cartesian moves get their own optional hook, ``_set_cartesian``, which defaults
to the joint path. Hardware overrides it so a Cartesian jog can go through the
robot's own IK -- which needs no servo calibration -- leaving only joint-space
rotation dependent on a measured pulse map.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..arm_state import ArmState, ArmTargetUnreachable
from ..calibration import CalibrationProfile
from ..contracts import (
    ARM_CARTESIAN_ACTIONS,
    ARM_ROTATION_ACTION,
    CHASSIS_TRANSLATION_ACTIONS,
    GRIPPER_ACTIONS,
    HOME_ARM_ACTION,
    ROTATION_ACTION,
    Action,
)
from ..kinematics import JointDegrees

LOG = logging.getLogger("bruno_core.vla.drivers")


class StrafeUnavailable(RuntimeError):
    """The chassis cannot translate sideways with its current wheels.

    Enforced here rather than in the hardware driver so simulation refuses it
    too. A sim that happily strafes a robot whose wheels cannot is no longer
    predicting anything -- it is the difference between a rehearsal and a
    daydream.
    """


@runtime_checkable
class MotionDriver(Protocol):
    def apply(self, action: Action) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...


class BaseDriver:
    name = "base"

    def __init__(self, profile: Optional[CalibrationProfile] = None):
        self.profile = profile or CalibrationProfile()
        self.arm = ArmState(self.profile)
        self.gripper_open = True
        self._lock = threading.RLock()
        self._last_action = Action.stop(f"{self.name} driver initialized")

    # ---------- subclass hooks ----------

    def _drive(self, vx_cmps: float, vy_cmps: float, wz_deg_s: float) -> None:
        raise NotImplementedError

    def _set_joints(self, joints_deg: JointDegrees, duration_ms: int) -> None:
        raise NotImplementedError

    def _set_cartesian(
        self,
        position: tuple,
        pitch_deg: float,
        joints_deg: JointDegrees,
        duration_ms: int,
    ) -> None:
        """Move to a solved Cartesian pose. Defaults to commanding the joints."""
        self._set_joints(joints_deg, duration_ms)

    def _set_gripper(self, opened: bool, duration_ms: int) -> None:
        raise NotImplementedError

    def _extra_status(self) -> Dict[str, Any]:
        return {}

    # ---------- dispatch ----------

    def apply(self, action: Action) -> None:
        if action.action == "stop":
            self.stop()
            return
        with self._lock:
            if action.action in CHASSIS_TRANSLATION_ACTIONS:
                self._apply_translation(action)
            elif action.action == ROTATION_ACTION:
                self._apply_rotation(action)
            elif action.action in ARM_CARTESIAN_ACTIONS:
                with self._arm_transaction():
                    self._apply_cartesian(self.arm.jog_cartesian(action), action)
            elif action.action == ARM_ROTATION_ACTION:
                with self._arm_transaction():
                    self._set_joints(
                        self.arm.rotate_joint(action.joint, action.angle_deg),
                        self._arm_duration(action),
                    )
            elif action.action == HOME_ARM_ACTION:
                with self._arm_transaction():
                    self._apply_cartesian(self.arm.home(), action)
            elif action.action in GRIPPER_ACTIONS:
                opened = action.action == "open_gripper"
                self._set_gripper(opened, self._arm_duration(action))
                self.gripper_open = opened
            else:
                raise RuntimeError(f"{self.name} driver does not implement {action.action}")
            self._last_action = action

    def _apply_translation(self, action: Action) -> None:
        if action.action in ("left", "right") and not self.profile.chassis.supports_strafe:
            raise StrafeUnavailable(
                "this chassis cannot strafe: its wheel rollers are mounted parallel "
                "rather than in an X, so a sideways command turns the robot instead "
                "of sliding it. Re-mount the wheels (front-left and rear-right share "
                "one roller direction, front-right and rear-left the other), then set "
                "vla.chassis.supports_strafe back to true"
            )
        speed_cmps = self.profile.chassis.linear_cmps(action.speed)
        # +x is the robot's right, +y is straight ahead, matching the arm frame.
        vx, vy = {
            "up": (0.0, speed_cmps),
            "down": (0.0, -speed_cmps),
            "left": (-speed_cmps, 0.0),
            "right": (speed_cmps, 0.0),
        }[action.action]
        self._drive(vx, vy, 0.0)

    def _apply_rotation(self, action: Action) -> None:
        # Positive angle_deg is counter-clockwise, so the rate carries its sign.
        rate = self.profile.chassis.rotation_deg_per_s(action.speed)
        direction = 1.0 if action.angle_deg >= 0 else -1.0
        self._drive(0.0, 0.0, direction * rate)

    @contextlib.contextmanager
    def _arm_transaction(self):
        """Roll the commanded arm state back if the driver refuses the move.

        `ArmState` has to be updated before the joint targets exist to be sent,
        so a hook that then raises -- an uncalibrated joint, a pose the robot's
        IK rejects -- would leave our model of the arm ahead of the arm itself.
        On hardware there is no feedback to notice the drift, and every later
        command would be computed from a pose the robot never reached.
        """
        saved = (list(self.arm.position), self.arm.pitch_deg, dict(self.arm.joints))
        try:
            yield
        except Exception:
            self.arm.position, self.arm.pitch_deg, self.arm.joints = (
                saved[0], saved[1], saved[2],
            )
            raise

    def _apply_cartesian(self, joints: JointDegrees, action: Action) -> None:
        self._set_cartesian(
            tuple(self.arm.position),
            self.arm.pitch_deg,
            joints,
            self._arm_duration(action),
        )

    def _arm_duration(self, action: Action) -> int:
        return action.duration_ms or self.profile.arm.move_time_ms

    # ---------- lifecycle ----------

    def stop(self) -> None:
        with self._lock:
            self._drive(0.0, 0.0, 0.0)
            self._last_action = Action.stop(f"{self.name} driver stopped")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            result: Dict[str, Any] = {
                "driver": self.name,
                "last_action": self._last_action.to_dict(),
                "arm": self.arm.snapshot(),
                "gripper": "open" if self.gripper_open else "closed",
            }
            result.update(self._extra_status())
            return result

    def close(self) -> None:
        pass


class MockDriver(BaseDriver):
    """Records commands and moves nothing. Used for tests and dry runs."""

    name = "mock"

    def __init__(self, profile: Optional[CalibrationProfile] = None):
        super().__init__(profile)
        self.command_count = 0
        self.velocity = (0.0, 0.0, 0.0)
        self.commanded_joints: JointDegrees = self.arm.joint_tuple()

    def _drive(self, vx_cmps: float, vy_cmps: float, wz_deg_s: float) -> None:
        self.velocity = (vx_cmps, vy_cmps, wz_deg_s)
        self.command_count += 1

    def _set_joints(self, joints_deg: JointDegrees, duration_ms: int) -> None:
        self.commanded_joints = joints_deg
        self.command_count += 1

    def _set_gripper(self, opened: bool, duration_ms: int) -> None:
        self.command_count += 1

    def _extra_status(self) -> Dict[str, Any]:
        vx, vy, wz = self.velocity
        return {
            "command_count": self.command_count,
            "velocity": {
                "vx_cmps": round(vx, 3),
                "vy_cmps": round(vy, 3),
                "wz_deg_s": round(wz, 3),
            },
        }


__all__ = [
    "ArmTargetUnreachable",
    "BaseDriver",
    "MockDriver",
    "MotionDriver",
    "StrafeUnavailable",
]
