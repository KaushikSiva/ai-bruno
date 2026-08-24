"""Commanded arm state, shared verbatim by the simulator and the hardware.

Both drivers own the same object and feed it the same actions, so a command
resolves to one joint vector regardless of which backend executes it. That is
what makes a sim run predictive of a hardware run rather than merely similar:
the two differ only in how the final joint angles reach an actuator, never in
how they were computed.

State is *commanded*, not measured. The simulator can read true joint positions
back out of the model; the MasterPi's PWM servos have no feedback path at all,
so on hardware this is the only estimate available.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from .calibration import CalibrationProfile
from .contracts import Action
from .kinematics import (
    JOINT_NAMES,
    JointDegrees,
    clamp_joint_deg,
    forward_kinematics,
    reachable_joint_degrees,
)

LOG = logging.getLogger("bruno_core.vla.arm_state")

# A Cartesian jog after a joint rotation can land on the other elbow branch,
# because IK always returns elbow-up for a given pose. Past this much sudden
# joint travel we warn rather than silently reconfiguring the arm.
BRANCH_JUMP_WARN_DEG = 45.0


class ArmTargetUnreachable(RuntimeError):
    """The requested arm target has no solution inside the joint travel."""


class ArmState:
    def __init__(self, profile: CalibrationProfile):
        self.profile = profile
        self.position = list(profile.arm.home_position)
        self.pitch_deg = profile.arm.home_pitch_deg
        solution = reachable_joint_degrees(*self.position, self.pitch_deg)
        if solution is None:
            raise ArmTargetUnreachable(
                f"configured arm home {tuple(self.position)} at pitch "
                f"{self.pitch_deg} deg is unreachable"
            )
        self.pitch_deg, joints = solution
        self.joints: Dict[str, float] = dict(zip(JOINT_NAMES, joints))

    # ---------- queries ----------

    def joint_tuple(self) -> JointDegrees:
        return tuple(self.joints[name] for name in JOINT_NAMES)  # type: ignore[return-value]

    def tcp_estimate(self) -> Tuple[Tuple[float, float, float], float]:
        return forward_kinematics(self.joint_tuple())

    def snapshot(self) -> Dict[str, object]:
        (x, y, z), pitch = self.tcp_estimate()
        return {
            "target_cm": {
                "x": round(self.position[0], 3),
                "y": round(self.position[1], 3),
                "z": round(self.position[2], 3),
                "pitch_deg": round(self.pitch_deg, 2),
            },
            "tcp_estimate_cm": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)},
            "tcp_pitch_deg": round(pitch, 2),
            "joints_deg": {name: round(value, 2) for name, value in self.joints.items()},
        }

    # ---------- commands ----------

    def home(self) -> JointDegrees:
        arm = self.profile.arm
        return self._set_cartesian(list(arm.home_position), arm.home_pitch_deg)

    def jog_cartesian(self, action: Action) -> JointDegrees:
        """Nudge the tool along one Hiwonder axis and re-solve the joints."""
        step = self.profile.arm.step_cm * max(0.1, min(1.0, action.speed))
        target = list(self.position)
        # x is lateral (+x is the robot's right), +y is straight ahead, z is up.
        deltas = {
            "move_arm_up": (2, +step),
            "move_arm_down": (2, -step),
            "move_arm_left": (0, -step),
            "move_arm_right": (0, +step),
            "move_arm_forward": (1, +step),
            "move_arm_backward": (1, -step),
        }
        axis, delta = deltas[action.action]
        target[axis] += delta
        return self._set_cartesian(list(self.profile.arm.clamp(target)), self.pitch_deg)

    def rotate_joint(self, joint: str, angle_deg: float) -> JointDegrees:
        """Turn one joint by a signed angle, then refresh the Cartesian pose."""
        if joint not in self.joints:
            raise ArmTargetUnreachable(f"unknown joint: {joint}")
        limit = self.profile.arm.max_joint_step_deg
        step = max(-limit, min(limit, float(angle_deg)))
        before = self.joints[joint]
        after = clamp_joint_deg(before + step)
        if abs(after - before) < 1e-9:
            raise ArmTargetUnreachable(
                f"{joint} is already at its {before:+.1f} deg travel limit"
            )
        self.joints[joint] = after
        (x, y, z), pitch = self.tcp_estimate()
        self.position = [x, y, z]
        self.pitch_deg = pitch
        return self.joint_tuple()

    # ---------- internals ----------

    def _set_cartesian(self, target: list, preferred_pitch: float) -> JointDegrees:
        solution = reachable_joint_degrees(target[0], target[1], target[2], preferred_pitch)
        if solution is None:
            raise ArmTargetUnreachable(
                f"no reachable solution for arm target "
                f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) cm"
            )
        pitch, joints = solution
        self._warn_on_branch_jump(joints)
        self.position = target
        self.pitch_deg = pitch
        self.joints = dict(zip(JOINT_NAMES, joints))
        return joints

    def _warn_on_branch_jump(self, joints: JointDegrees) -> None:
        jump = max(
            abs(new - self.joints[name]) for name, new in zip(JOINT_NAMES, joints)
        )
        if jump > BRANCH_JUMP_WARN_DEG:
            LOG.warning(
                "Cartesian jog reconfigures the arm by %.0f deg; IK only returns "
                "the elbow-up branch, so a preceding joint rotation may be undone",
                jump,
            )


def make_arm_state(profile: Optional[CalibrationProfile] = None) -> ArmState:
    return ArmState(profile or CalibrationProfile())
