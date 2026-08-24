"""The one place sim and hardware agree on what a command physically means.

Every number the two drivers need in order to produce the *same* motion lives
here. Commands are expressed in physical units -- cm/s, deg/s, degrees, cm --
and each driver converts to its own actuator units on the way out:

    speed 0.6  ->  0.6 * max_linear_speed_cmps  ->  m/s for a MuJoCo velocity
                                               ->  set_velocity units on hardware

    joint +15 deg -> a MuJoCo hinge target in radians
                  -> a PWM pulse on the matching MasterPi channel

That indirection is the whole point: calibrating one physical number moves sim
and robot together instead of letting them drift apart.

Servo defaults follow the pinned MJCF's own generator, which states
``500..2500 us = 0..180 deg, so every arm joint ranges +/- 90 deg``. Direction
and centre, though, are per-robot facts that cannot be derived from a model, so
each joint carries a ``verified`` flag. Hardware refuses joint-space commands
on an unverified joint unless explicitly overridden -- a wrong sign there does
not mean a small error, it means the joint slams to the opposite limit. Run
``bruno_apps/teleop/calibrate_joints.py`` on the robot to fill them in.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

from .kinematics import JOINT_LIMIT_DEG, JOINT_NAMES, JOINT_SERVO_CHANNELS, clamp_joint_deg

# 500..2500 us spans the servo's full 180 deg of travel.
DEFAULT_PULSE_PER_DEGREE = 2000.0 / 180.0
DEGREES_PER_RADIAN = 180.0 / math.pi


@dataclass(frozen=True)
class ServoCalibration:
    """Maps one arm joint angle in degrees to a PWM pulse on its channel.

    ``deviation_us`` is the robot's own stored servo trim, read from the
    MasterPi SDK's ``deviation_data``. Hiwonder's ``ArmIK.servosMove`` adds it
    to every pulse it writes, so anything driving the servos directly has to as
    well or it lands somewhere else entirely -- this robot's base trim is -95
    us, which is 8.5 degrees of error at the shoulder of the arm.

    Unlike the rest of this profile it is a property of one physical robot, not
    of the MasterPi design. Copying a config to a second Bruno means re-reading
    its deviations; everything else transfers.
    """

    channel: int
    center_pulse: float = 1500.0
    pulse_per_degree: float = DEFAULT_PULSE_PER_DEGREE
    sign: int = 1
    deviation_us: float = 0.0
    min_pulse: int = 500
    max_pulse: int = 2500
    verified: bool = False

    def pulse_for(self, joint_deg: float) -> int:
        raw = (
            self.center_pulse
            + self.deviation_us
            + self.sign * clamp_joint_deg(joint_deg) * self.pulse_per_degree
        )
        return int(round(max(self.min_pulse, min(self.max_pulse, raw))))

    def joint_deg_for(self, pulse: float) -> float:
        if self.pulse_per_degree == 0:
            return 0.0
        offset = float(pulse) - self.center_pulse - self.deviation_us
        return offset / (self.sign * self.pulse_per_degree)

    def reachable_deg(self) -> Tuple[float, float]:
        """Travel actually available once the trim is applied.

        A non-zero trim shifts the whole range, so it costs travel at one end.
        Worth knowing because Hiwonder's own IK does not: it range-checks the
        pulse *before* servosMove adds the trim, and will happily write past the
        servo's limit at the far end of a joint's travel. We clamp instead, and
        this is the honest statement of what that leaves.
        """
        if self.pulse_per_degree == 0:
            return (0.0, 0.0)
        edges = sorted(
            (self.joint_deg_for(self.min_pulse), self.joint_deg_for(self.max_pulse))
        )
        return (
            max(-JOINT_LIMIT_DEG, edges[0]),
            min(JOINT_LIMIT_DEG, edges[1]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "center_pulse": round(self.center_pulse, 3),
            "pulse_per_degree": round(self.pulse_per_degree, 5),
            "sign": self.sign,
            "deviation_us": round(self.deviation_us, 3),
            "min_pulse": self.min_pulse,
            "max_pulse": self.max_pulse,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class ChassisCalibration:
    """Chassis speeds in physical units, plus each backend's unit conversion."""

    # Physical truth at action speed 1.0. Measure these on the floor.
    max_linear_speed_cmps: float = 30.0
    max_rotation_deg_per_s: float = 90.0
    # MasterPi's mecanum.set_velocity takes mm/s and rad/s, hence 10 and pi/180.
    real_velocity_units_per_cmps: float = 10.0
    real_rotation_units_per_deg_per_s: float = math.pi / 180.0
    # Strafing needs the wheel rollers to form an X seen from above. Mounted
    # parallel, the sideways components make a torque instead of a translation
    # and the robot spins when asked to slide. Set false and left/right are
    # refused rather than quietly turning the robot.
    supports_strafe: bool = True

    def linear_cmps(self, speed: float) -> float:
        return max(0.0, min(1.0, speed)) * self.max_linear_speed_cmps

    def rotation_deg_per_s(self, speed: float) -> float:
        return max(0.0, min(1.0, speed)) * self.max_rotation_deg_per_s

    def rotation_duration_ms(self, angle_deg: float, speed: float) -> int:
        """How long to spin to cover ``angle_deg`` -- identical in sim and real."""
        rate = self.rotation_deg_per_s(max(0.05, speed))
        if rate <= 0.0:
            return 0
        return int(math.ceil(abs(angle_deg) / rate * 1000.0))


@dataclass(frozen=True)
class ArmCalibration:
    """Home pose, workspace box, and jog sizes, all in the Hiwonder frame."""

    home_position: tuple = (0.0, 6.0, 18.0)
    home_pitch_deg: float = 0.0
    step_cm: float = 2.0
    min_x: float = -10.0
    max_x: float = 10.0
    min_y: float = 3.0
    max_y: float = 20.0
    min_z: float = 4.0
    max_z: float = 24.0
    joint_step_deg: float = 5.0
    max_joint_step_deg: float = 30.0
    move_time_ms: int = 600

    def clamp(self, position) -> tuple:
        x, y, z = position
        return (
            max(self.min_x, min(self.max_x, float(x))),
            max(self.min_y, min(self.max_y, float(y))),
            max(self.min_z, min(self.max_z, float(z))),
        )


@dataclass(frozen=True)
class GripperCalibration:
    channel: int = 1
    open_pulse: int = 2000
    closed_pulse: int = 1500
    sim_open_m: float = 0.020
    sim_closed_m: float = 0.002

    def pulse_for(self, opened: bool) -> int:
        return self.open_pulse if opened else self.closed_pulse

    def opening_m_for(self, opened: bool) -> float:
        return self.sim_open_m if opened else self.sim_closed_m


@dataclass(frozen=True)
class CalibrationProfile:
    """Everything a driver needs to turn an action into physical motion."""

    chassis: ChassisCalibration = field(default_factory=ChassisCalibration)
    arm: ArmCalibration = field(default_factory=ArmCalibration)
    gripper: GripperCalibration = field(default_factory=GripperCalibration)
    servos: Dict[str, ServoCalibration] = field(
        default_factory=lambda: {
            name: ServoCalibration(channel=JOINT_SERVO_CHANNELS[name])
            for name in JOINT_NAMES
        }
    )

    # ---------- unit conversions ----------

    def sim_linear_mps(self, speed: float) -> float:
        return self.chassis.linear_cmps(speed) / 100.0

    def sim_rotation_rad_s(self, speed: float) -> float:
        return math.radians(self.chassis.rotation_deg_per_s(speed))

    def real_velocity_units(self, speed: float) -> float:
        return self.chassis.linear_cmps(speed) * self.chassis.real_velocity_units_per_cmps

    def real_rotation_units(self, speed: float) -> float:
        return (
            self.chassis.rotation_deg_per_s(speed)
            * self.chassis.real_rotation_units_per_deg_per_s
        )

    def unverified_joints(self) -> tuple:
        return tuple(name for name in JOINT_NAMES if not self.servos[name].verified)

    # ---------- persistence ----------

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CalibrationProfile":
        block = dict(data or {})
        defaults = cls()

        chassis_raw = block.get("chassis") or {}
        chassis_kwargs: Dict[str, Any] = {
            key: float(chassis_raw[key])
            for key in (
                "max_linear_speed_cmps",
                "max_rotation_deg_per_s",
                "real_velocity_units_per_cmps",
                "real_rotation_units_per_deg_per_s",
            )
            if key in chassis_raw
        }
        if "supports_strafe" in chassis_raw:
            chassis_kwargs["supports_strafe"] = bool(chassis_raw["supports_strafe"])
        chassis = replace(defaults.chassis, **chassis_kwargs)

        arm_raw = block.get("arm") or {}
        arm_kwargs: Dict[str, Any] = {}
        if "home_position" in arm_raw:
            home = arm_raw["home_position"]
            arm_kwargs["home_position"] = (float(home[0]), float(home[1]), float(home[2]))
        for key in (
            "home_pitch_deg", "step_cm", "min_x", "max_x", "min_y",
            "max_y", "min_z", "max_z", "joint_step_deg", "max_joint_step_deg",
        ):
            if key in arm_raw:
                arm_kwargs[key] = float(arm_raw[key])
        if "move_time_ms" in arm_raw:
            arm_kwargs["move_time_ms"] = int(arm_raw["move_time_ms"])
        arm = replace(defaults.arm, **arm_kwargs)

        gripper_raw = block.get("gripper") or {}
        gripper = replace(
            defaults.gripper,
            **{
                key: (float(gripper_raw[key]) if key.startswith("sim_") else int(gripper_raw[key]))
                for key in ("channel", "open_pulse", "closed_pulse", "sim_open_m", "sim_closed_m")
                if key in gripper_raw
            },
        )

        servos_raw = block.get("servos") or {}
        servos = {}
        for name in JOINT_NAMES:
            entry = servos_raw.get(name) or {}
            servos[name] = ServoCalibration(
                channel=int(entry.get("channel", JOINT_SERVO_CHANNELS[name])),
                center_pulse=float(entry.get("center_pulse", 1500.0)),
                pulse_per_degree=float(entry.get("pulse_per_degree", DEFAULT_PULSE_PER_DEGREE)),
                sign=1 if int(entry.get("sign", 1)) >= 0 else -1,
                deviation_us=float(entry.get("deviation_us", 0.0)),
                min_pulse=int(entry.get("min_pulse", 500)),
                max_pulse=int(entry.get("max_pulse", 2500)),
                verified=bool(entry.get("verified", False)),
            )
        return cls(chassis=chassis, arm=arm, gripper=gripper, servos=servos)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chassis": {
                "max_linear_speed_cmps": self.chassis.max_linear_speed_cmps,
                "max_rotation_deg_per_s": self.chassis.max_rotation_deg_per_s,
                "real_velocity_units_per_cmps": self.chassis.real_velocity_units_per_cmps,
                "real_rotation_units_per_deg_per_s": self.chassis.real_rotation_units_per_deg_per_s,
                "supports_strafe": self.chassis.supports_strafe,
            },
            "arm": {
                "home_position": list(self.arm.home_position),
                "home_pitch_deg": self.arm.home_pitch_deg,
                "step_cm": self.arm.step_cm,
                "min_x": self.arm.min_x, "max_x": self.arm.max_x,
                "min_y": self.arm.min_y, "max_y": self.arm.max_y,
                "min_z": self.arm.min_z, "max_z": self.arm.max_z,
                "joint_step_deg": self.arm.joint_step_deg,
                "max_joint_step_deg": self.arm.max_joint_step_deg,
                "move_time_ms": self.arm.move_time_ms,
            },
            "gripper": {
                "channel": self.gripper.channel,
                "open_pulse": self.gripper.open_pulse,
                "closed_pulse": self.gripper.closed_pulse,
                "sim_open_m": self.gripper.sim_open_m,
                "sim_closed_m": self.gripper.sim_closed_m,
            },
            "servos": {name: servo.to_dict() for name, servo in self.servos.items()},
        }


def fit_servo_calibration(
    channel: int,
    samples: Sequence[Tuple[float, float]],
    max_residual_us: float = 2.0,
) -> Tuple[ServoCalibration, float]:
    """Least-squares fit ``pulse = center + sign * ppd * joint_deg`` from samples.

    ``samples`` are ``(joint_deg, pulse_us)`` pairs. Returns the calibration and
    the worst residual in microseconds; the joint is marked verified only when
    the fit actually explains the data, since a poor fit means the relationship
    is not the linear one this model assumes and the numbers should not be
    trusted to drive a servo.
    """
    points = [(float(angle), float(pulse)) for angle, pulse in samples]
    if len(points) < 2:
        raise ValueError("fitting a servo calibration needs at least two samples")
    count = len(points)
    mean_angle = sum(angle for angle, _ in points) / count
    mean_pulse = sum(pulse for _, pulse in points) / count
    variance = sum((angle - mean_angle) ** 2 for angle, _ in points)
    if variance < 1e-9:
        raise ValueError("all samples share one joint angle; vary the pose")
    slope = sum(
        (angle - mean_angle) * (pulse - mean_pulse) for angle, pulse in points
    ) / variance
    intercept = mean_pulse - slope * mean_angle
    if abs(slope) < 1e-6:
        raise ValueError("the fitted pulse does not vary with joint angle")
    worst = max(abs(slope * angle + intercept - pulse) for angle, pulse in points)
    calibration = ServoCalibration(
        channel=channel,
        center_pulse=intercept,
        pulse_per_degree=abs(slope),
        sign=1 if slope > 0 else -1,
        verified=worst <= max_residual_us,
    )
    return calibration, worst


def repo_root() -> str:
    """Repository root, derived from this file so it works from any cwd."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def config_path() -> str:
    return os.path.join(repo_root(), "config", "bruno_config.json")


def load_profile(path: str = "") -> CalibrationProfile:
    """Read the ``vla`` block of bruno_config.json, falling back to defaults."""
    target = path or config_path()
    try:
        with open(target, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return CalibrationProfile()
    return CalibrationProfile.from_dict((config or {}).get("vla"))


def save_profile(profile: CalibrationProfile, path: str = "") -> str:
    """Write the profile back into the ``vla`` block, leaving the rest intact."""
    target = path or config_path()
    try:
        with open(target, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        config = {}
    config["vla"] = profile.to_dict()
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")
    return target
