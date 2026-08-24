#!/usr/bin/env python3
"""
Derive Bruno's arm servo pulse map, so joint-space commands are safe to send.

`rotate_arm_by_deg` writes PWM pulses directly, which means it needs to know
each joint's centre pulse, microseconds per degree, and direction. Guessing the
direction wrong does not produce a small error -- it drives the joint to the
opposite limit -- so `bruno_core.vla` refuses joint-space commands until this
script has run.

The robot already knows the answer. Hiwonder's `ArmIK.transformAngelAdaptArm`
is the conversion its own IK uses to turn joint angles into pulses, so this
script reads the map straight out of the SDK rather than asking anyone to eyeball
a protractor. Nothing moves: the SDK call is pure arithmetic.

Run it on the robot:

    python3 bruno_apps/teleop/calibrate_joints.py            # inspect the fit
    python3 bruno_apps/teleop/calibrate_joints.py --write    # save it

Results land in the `vla.servos` block of config/bruno_config.json.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.vla.calibration import (
    CalibrationProfile,
    fit_servo_calibration,
    load_profile,
    save_profile,
)
from bruno_core.vla.kinematics import (
    JOINT_NAMES,
    JOINT_SERVO_CHANNELS,
    hiwonder_ik,
    servo_angles_to_joint_degrees,
)

# Poses spread across the workspace so every joint sweeps a useful range. A
# joint that barely moves across the set cannot be fitted from it.
SAMPLE_POSES: Tuple[Tuple[float, float, float, float], ...] = (
    (0, 6, 18, 0), (0, 8, 18, 0), (0, 10, 16, 0), (0, 12, 12, 0),
    (0, 14, 8, -20), (0, 15, 6, -30), (0, 10, 20, 20), (0, 8, 22, 30),
    (4, 8, 18, 0), (8, 10, 14, 0), (-4, 8, 18, 0), (-8, 10, 14, 0),
    (6, 12, 10, -20), (-6, 12, 10, -20), (2, 7, 21, 15), (-2, 7, 21, 15),
)


def load_arm_ik():
    """Import the robot's own IK, or explain why it is not available."""
    sys.path.append(os.getenv("BRUNO_MASTERPI_PATH", "/home/pi/MasterPi"))
    try:
        from kinematics.arm_move_ik import ArmIK  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"Could not import the MasterPi SDK ({exc}).\n"
            "This script derives the pulse map from the robot's own IK, so it "
            "has to run on Bruno. Set BRUNO_MASTERPI_PATH if the SDK lives "
            "somewhere other than /home/pi/MasterPi."
        ) from exc
    arm_ik = ArmIK()
    if not hasattr(arm_ik, "transformAngelAdaptArm"):
        raise SystemExit(
            "This MasterPi SDK build has no ArmIK.transformAngelAdaptArm, so the "
            "pulse map cannot be read from it. Calibrate by hand instead: jog each "
            "joint to two known angles, record the pulses, and write them into the "
            "vla.servos block of config/bruno_config.json."
        )
    return arm_ik


def collect_samples(arm_ik) -> Dict[str, List[Tuple[float, float]]]:
    """Pair each joint angle with the pulse the SDK would send for it."""
    samples: Dict[str, List[Tuple[float, float]]] = {name: [] for name in JOINT_NAMES}
    skipped = 0
    for x, y, z, pitch in SAMPLE_POSES:
        solution = hiwonder_ik(x, y, z, pitch)
        if solution is None:
            skipped += 1
            continue
        theta3, theta4, theta5, theta6 = solution
        pulses = arm_ik.transformAngelAdaptArm(theta3, theta4, theta5, theta6)
        if not pulses:
            skipped += 1
            continue
        joints = servo_angles_to_joint_degrees(theta3, theta4, theta5, theta6)
        for name, angle in zip(JOINT_NAMES, joints):
            pulse = pulses.get(f"servo{JOINT_SERVO_CHANNELS[name]}")
            if pulse is not None:
                samples[name].append((angle, float(pulse)))
    if skipped:
        print(f"  ({skipped} of {len(SAMPLE_POSES)} sample poses had no IK solution)")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive Bruno's arm servo pulse map")
    parser.add_argument("--write", action="store_true",
                        help="Save the result into config/bruno_config.json")
    parser.add_argument("--config", default="", help="Path to bruno_config.json")
    parser.add_argument("--max-residual-us", type=float, default=2.0,
                        help="Worst acceptable fit error before a joint stays unverified")
    args = parser.parse_args()

    arm_ik = load_arm_ik()
    print("Reading the pulse map from the robot's own IK (nothing will move)\n")
    samples = collect_samples(arm_ik)

    profile = load_profile(args.config)
    servos = dict(profile.servos)
    print(f"{'joint':>10} {'samples':>8} {'centre':>9} {'us/deg':>8} {'dir':>5} {'worst err':>10}  status")
    all_verified = True
    for name in JOINT_NAMES:
        points = samples[name]
        try:
            calibration, worst = fit_servo_calibration(
                JOINT_SERVO_CHANNELS[name], points, args.max_residual_us
            )
        except ValueError as exc:
            all_verified = False
            print(f"{name:>10} {len(points):>8} {'-':>9} {'-':>8} {'-':>5} {'-':>10}  FAILED: {exc}")
            continue
        servos[name] = calibration
        all_verified = all_verified and calibration.verified
        direction = "+" if calibration.sign > 0 else "-"
        status = "verified" if calibration.verified else f"NOT verified (> {args.max_residual_us} us)"
        print(
            f"{name:>10} {len(points):>8} {calibration.center_pulse:>9.1f} "
            f"{calibration.pulse_per_degree:>8.4f} {direction:>5} {worst:>9.2f}us  {status}"
        )

    updated = CalibrationProfile(
        chassis=profile.chassis, arm=profile.arm, gripper=profile.gripper, servos=servos
    )
    if not args.write:
        print("\nNothing written. Re-run with --write to save this into the config.")
        return 0
    path = save_profile(updated, args.config)
    print(f"\nSaved to {path}")
    if all_verified:
        print("All four joints verified: rotate_arm_by_deg is now enabled on hardware.")
        print("Test with the arm clear, one small step at a time:")
        print("  python3 bruno_apps/teleop/main.py --target real rotate_arm_by_deg --joint base --angle 5")
    else:
        print("Some joints did not fit cleanly and stay disabled. Check the residuals above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
