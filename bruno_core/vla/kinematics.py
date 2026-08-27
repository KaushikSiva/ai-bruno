"""MasterPi arm kinematics and the sim/real angle conversions.

Pure math: nothing here imports MuJoCo or the MasterPi SDK, so both drivers and
the tests can share it off-robot.

Two frames are in play and they are easy to confuse.

*Hiwonder frame* is what the robot's own ``ArmIK.setPitchRangeMoving`` takes:
``x`` is lateral, **+y is straight ahead**, ``z`` is up, all in cm, measured
from the arm's IK origin. Hiwonder's own examples home the arm at ``(0, 6, 18)``.

*Joint space* is the four hinges as MuJoCo models them, in degrees: ``base``
yaw about z, then ``shoulder``, ``elbow``, ``wrist`` pitching in the arm plane.
Each is clamped to +/-90 deg, which is the travel the MJCF and the servos agree
on. ``shoulder = 0`` is the arm pointing straight up.

The conversion between the two is transcribed from ``verify_fk.py`` in the
pinned ``third_party/masterpi-mujoco`` model, which checks itself against the
stock ``ArmIK/InverseKinematics.py`` on the robot image. Keeping it in one
place is what lets a joint command mean the same thing in sim and on hardware.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# cm. From ArmIK/InverseKinematics.py on the robot; ArmMoveIK adds 1.3 to l1.
L1_CM = 9.30
L2_CM = 6.50
L3_CM = 6.20
L4_CM = 10.00

# Order matters: this is the canonical joint ordering used everywhere below.
JOINT_NAMES: Tuple[str, str, str, str] = ("base", "shoulder", "elbow", "wrist")

# PWM channel on the MasterPi board driving each joint, and the MJCF actuator
# driving it in simulation. Channels 3-6 are the arm; 1 is the gripper and 2 is
# the camera pan servo, neither of which belongs to the kinematic chain.
JOINT_SERVO_CHANNELS = {"base": 6, "shoulder": 5, "elbow": 4, "wrist": 3}
JOINT_ACTUATORS = {
    "base": "servo6_base",
    "shoulder": "servo5_shoulder",
    "elbow": "servo4_elbow",
    "wrist": "servo3_wrist",
}

JOINT_LIMIT_DEG = 90.0

JointDegrees = Tuple[float, float, float, float]
Pose = Tuple[float, float, float]


def clamp_joint_deg(value: float) -> float:
    """Clamp one joint angle to the travel the MJCF and the servos share."""
    return max(-JOINT_LIMIT_DEG, min(JOINT_LIMIT_DEG, float(value)))


def hiwonder_ik(
    x_cm: float,
    y_cm: float,
    z_cm: float,
    pitch_deg: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Hiwonder's own IK, returning ``(theta3, theta4, theta5, theta6)`` in deg.

    Returns None when the target is out of reach. This is deliberately a
    transcription rather than an improvement: the point is to agree with what
    the robot's on-board Python computes for the same target.
    """
    theta6 = math.degrees(math.atan2(y_cm, x_cm))
    reach_plane = math.hypot(x_cm, y_cm)
    tool_radial = L4_CM * math.cos(math.radians(pitch_deg))
    tool_vertical = L4_CM * math.sin(math.radians(pitch_deg))
    wrist_radial = reach_plane - tool_radial
    wrist_height = z_cm - L1_CM - tool_vertical
    wrist_reach = math.hypot(wrist_radial, wrist_height)
    if wrist_reach < 1e-9 or L2_CM + L3_CM < round(wrist_reach, 4):
        return None

    cos_elbow = round((L2_CM**2 + L3_CM**2 - wrist_reach**2) / (2 * L2_CM * L3_CM), 4)
    if abs(cos_elbow) > 1:
        return None
    theta4 = 180.0 - math.degrees(math.acos(cos_elbow))

    cos_shoulder = round(
        (wrist_reach**2 + L2_CM**2 - L3_CM**2) / (2 * L2_CM * wrist_reach), 4
    )
    if abs(cos_shoulder) > 1:
        return None
    elevation = math.acos(max(-1.0, min(1.0, wrist_radial / wrist_reach)))
    theta5 = math.degrees(
        elevation * (-1 if wrist_height < 0 else 1) + math.acos(cos_shoulder)
    )
    theta3 = pitch_deg - theta5 + theta4
    return theta3, theta4, theta5, theta6


def servo_angles_to_joint_degrees(
    theta3: float,
    theta4: float,
    theta5: float,
    theta6: float,
) -> JointDegrees:
    """Hiwonder IK output -> canonical joint degrees, in ``JOINT_NAMES`` order."""
    # Hiwonder measures forward as +y; the model's base zero points along +x.
    return (theta6 - 90.0, 90.0 - theta5, theta4, -theta3)


def joint_degrees_to_servo_angles(joints: JointDegrees) -> Tuple[float, float, float, float]:
    """Inverse of :func:`servo_angles_to_joint_degrees`, as ``(t3, t4, t5, t6)``."""
    base, shoulder, elbow, wrist = joints
    return (-wrist, elbow, 90.0 - shoulder, base + 90.0)


def ik_joint_degrees(
    x_cm: float,
    y_cm: float,
    z_cm: float,
    pitch_deg: float = 0.0,
) -> Optional[JointDegrees]:
    """Cartesian Hiwonder target -> joint degrees, or None if unreachable."""
    solution = hiwonder_ik(x_cm, y_cm, z_cm, pitch_deg)
    if solution is None:
        return None
    return servo_angles_to_joint_degrees(*solution)


def reachable_joint_degrees(
    x_cm: float,
    y_cm: float,
    z_cm: float,
    preferred_pitch_deg: float = 0.0,
) -> Optional[Tuple[float, JointDegrees]]:
    """Find the pitch nearest ``preferred_pitch_deg`` that stays inside travel.

    A Cartesian target can be geometrically reachable at one tool pitch and
    outside a joint limit at another, so sweeping outward from the preferred
    pitch keeps incremental Cartesian moves from dead-ending. Returns the pitch
    actually used alongside its joint solution.
    """
    for offset in range(181):
        for candidate in {
            max(-90.0, min(90.0, preferred_pitch_deg - offset)),
            max(-90.0, min(90.0, preferred_pitch_deg + offset)),
        }:
            joints = ik_joint_degrees(x_cm, y_cm, z_cm, candidate)
            if joints is None:
                continue
            if all(abs(angle) <= JOINT_LIMIT_DEG + 1e-6 for angle in joints):
                return candidate, joints
    return None


def forward_kinematics(joints: JointDegrees) -> Tuple[Pose, float]:
    """Joint degrees -> ``((x, y, z) cm in the Hiwonder frame, tool pitch deg)``.

    Used so the hardware driver can report a TCP estimate in the same units the
    simulator reads out of the model, which is what makes the two comparable.
    """
    base, shoulder, elbow, wrist = joints
    upper = math.radians(shoulder)
    fore = math.radians(shoulder + elbow)
    tool = math.radians(shoulder + elbow + wrist)
    radial = L2_CM * math.sin(upper) + L3_CM * math.sin(fore) + L4_CM * math.sin(tool)
    height = (
        L1_CM
        + L2_CM * math.cos(upper)
        + L3_CM * math.cos(fore)
        + L4_CM * math.cos(tool)
    )
    heading = math.radians(base + 90.0)
    pose = (radial * math.cos(heading), radial * math.sin(heading), height)
    # The tool axis sits 90 deg from the pitch Hiwonder quotes, and runs the
    # opposite way: substituting the IK's own definitions gives
    # shoulder + elbow + wrist == 90 - pitch exactly.
    return pose, 90.0 - math.degrees(tool)
