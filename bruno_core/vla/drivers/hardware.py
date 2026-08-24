"""Physical MasterPi driver, built on this repo's existing motion and arm code.

Reuses `bruno_core.motion.mecanum.MecanumWrapper` for the chassis and
`bruno_core.manipulation.arm.ArmController` for the board handle, its IK, and
its dry-run fallback, rather than opening a second, competing path to the same
hardware.

The arm has two routes, and which one a command takes matters:

*Cartesian* jogs and homing go through the robot's own
``ArmIK.setPitchRangeMoving``. Hiwonder's IK already knows how to turn a pose
into servo pulses, so this path needs no calibration and works out of the box.
It is verified to agree with the simulator's kinematics to well under a
millimetre.

*Joint-space* rotation (``rotate_arm_by_deg``) has no equivalent in Hiwonder's
API and must write pulses directly, which requires a measured pulse map. An
unverified map is not a small error -- a wrong sign drives the joint to the
opposite limit -- so those commands are refused until
``bruno_apps/teleop/calibrate_joints.py`` has filled the map in.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

from ..calibration import CalibrationProfile
from ..kinematics import JOINT_NAMES, JointDegrees
from .base import BaseDriver

LOG = logging.getLogger("bruno_core.vla.drivers.hardware")


class UncalibratedJoint(RuntimeError):
    """A joint-space command was issued against an unverified pulse map."""


class _DryChassis:
    """Stand-in used when the MasterPi SDK is missing, so teleop still runs."""

    def set_velocity(self, speed: float, direction_deg: float, rotation: float) -> None:
        LOG.info(
            "[dry] chassis set_velocity(%.1f, %.1f deg, %.3f)", speed, direction_deg, rotation
        )

    def stop(self) -> None:
        LOG.info("[dry] chassis stop")


class HardwareDriver(BaseDriver):
    name = "bruno"

    def __init__(
        self,
        profile: Optional[CalibrationProfile] = None,
        dry_run: bool = False,
        allow_uncalibrated: bool = False,
    ):
        super().__init__(profile)
        self.dry_run = dry_run
        self.allow_uncalibrated = allow_uncalibrated or os.getenv(
            "BRUNO_VLA_ALLOW_UNCALIBRATED", ""
        ).lower() in {"1", "true", "yes"}

        from bruno_core.manipulation.arm import ArmConfig, ArmController

        arm_cfg = ArmConfig()
        arm_cfg.move_time_ms = self.profile.arm.move_time_ms
        # The VLA layer owns pitch per command, so keep the controller neutral.
        arm_cfg.grasp_pitch = int(self.profile.arm.home_pitch_deg)
        arm_cfg.gripper_servo_id = self.profile.gripper.channel
        arm_cfg.gripper_open_pulse = self.profile.gripper.open_pulse
        arm_cfg.gripper_closed_pulse = self.profile.gripper.closed_pulse
        self.arm_controller = ArmController(cfg=arm_cfg, dry_run=dry_run)

        self.chassis = self._make_chassis(dry_run)
        unverified = self.profile.unverified_joints()
        if unverified:
            LOG.warning(
                "Servo pulse map unverified for: %s. Cartesian arm control works "
                "anyway (it uses the robot's own IK), but rotate_arm_by_deg is "
                "refused until you run bruno_apps/teleop/calibrate_joints.py",
                ", ".join(unverified),
            )

    def _make_chassis(self, dry_run: bool):
        if dry_run:
            LOG.info("Hardware driver in dry mode: the chassis will not move")
            return _DryChassis()
        try:
            from bruno_core.motion.mecanum import MecanumWrapper

            return MecanumWrapper()
        except Exception as exc:
            LOG.error("MasterPi chassis unavailable (%s); falling back to dry mode", exc)
            return _DryChassis()

    # ---------- BaseDriver hooks ----------

    def _drive(self, vx_cmps: float, vy_cmps: float, wz_deg_s: float) -> None:
        """Translate a body-frame velocity into MasterPi's set_velocity call.

        Hiwonder takes a speed magnitude, a heading in degrees where 0 is the
        robot's right and 90 is straight ahead, and an angular rate.

        That angular rate is positive counter-clockwise, the same convention
        this repo uses, so it passes through unchanged. This was measured on the
        floor, not assumed: an earlier negation here turned a commanded +45 into
        45 degrees to the right. Note that it also contradicts
        `bruno_core/motion/mecanum.py`, whose `turn_left` passes a negative rate
        and `turn_right` a positive one -- those two labels look swapped, but
        other apps may be built around them, so they are left alone.
        """
        speed_cmps = math.hypot(vx_cmps, vy_cmps)
        magnitude = speed_cmps * self.profile.chassis.real_velocity_units_per_cmps
        direction_deg = math.degrees(math.atan2(vy_cmps, vx_cmps)) % 360.0 if magnitude else 90.0
        rotation = wz_deg_s * self.profile.chassis.real_rotation_units_per_deg_per_s
        self.chassis.set_velocity(magnitude, direction_deg, rotation)

    def _set_cartesian(
        self, position: tuple, pitch_deg: float, joints_deg: JointDegrees, duration_ms: int
    ) -> None:
        ok = self.arm_controller.move_to(position, move_time_ms=duration_ms, pitch=int(pitch_deg))
        if not ok:
            raise RuntimeError(
                f"MasterPi IK rejected arm target "
                f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}) cm "
                f"at pitch {pitch_deg:.0f} deg"
            )

    def _set_joints(self, joints_deg: JointDegrees, duration_ms: int) -> None:
        unverified = self.profile.unverified_joints()
        if unverified and not self.allow_uncalibrated:
            raise UncalibratedJoint(
                f"joint-space arm commands need a verified pulse map; {', '.join(unverified)} "
                "unverified. Run bruno_apps/teleop/calibrate_joints.py, or pass "
                "--allow-uncalibrated if you accept the risk of a joint slamming to its limit"
            )
        targets: List[tuple] = []
        for name, angle in zip(JOINT_NAMES, joints_deg):
            servo = self.profile.servos[name]
            targets.append((servo.channel, servo.pulse_for(angle)))
        move_time = max(0.05, duration_ms / 1000.0)
        if not self.arm_controller.set_servos(targets, move_time):
            raise RuntimeError(f"servo write failed for {targets}")

    def _set_gripper(self, opened: bool, duration_ms: int) -> None:
        pulse = self.profile.gripper.pulse_for(opened)
        if not self.arm_controller.set_gripper(pulse, max(0.05, duration_ms / 1000.0)):
            raise RuntimeError(f"gripper write failed (pulse {pulse})")

    def _extra_status(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "arm_enabled": bool(getattr(self.arm_controller, "enabled", False)),
            "unverified_joints": list(self.profile.unverified_joints()),
            "commanded_pulses": {
                name: self.profile.servos[name].pulse_for(self.arm.joints[name])
                for name in JOINT_NAMES
            },
        }

    def stop(self) -> None:
        super().stop()
        try:
            self.chassis.stop()
        except Exception as exc:  # a failed stop must be visible, not swallowed
            LOG.error("Chassis stop failed: %s", exc)
            raise
