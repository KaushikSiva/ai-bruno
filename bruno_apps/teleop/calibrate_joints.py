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
from dataclasses import replace
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
    joint_degrees_to_servo_angles,
    servo_angles_to_joint_degrees,
)

# Sweep each joint across its travel with the others held neutral. Sampling
# Cartesian poses instead would be a mistake: the SDK refuses most of them, and
# the few it accepts can pin a joint inside a two-degree window, where the
# pulses' integer rounding swamps the slope being fitted.
SWEEP_DEGREES: Tuple[int, ...] = tuple(range(-85, 86, 5))

# Real poses, used only to check the fitted map against the SDK end to end.
CHECK_POSES: Tuple[Tuple[float, float, float, float], ...] = (
    (0, 6, 18, 0), (0, 8, 18, 0), (0, 10, 16, 0), (0, 12, 12, 0),
    (0, 15, 6, -30), (4, 8, 18, 0), (-4, 8, 18, 0), (2, 7, 21, 15),
    (-2, 7, 21, 15), (0, 10, 20, 20),
)


def load_deviations() -> Dict[str, float]:
    """Read the robot's stored per-servo trim, which servosMove adds to every pulse.

    Omitting this is not a rounding error: on the robot this was written for,
    the base trim is -95 us, so a pulse computed without it lands 8.5 degrees
    away from where the robot's own IK would have put it.
    """
    try:
        from kinematics.arm_move_ik import deviation_data  # type: ignore
    except Exception:
        print("  warning: could not read deviation_data; assuming zero trim")
        return {name: 0.0 for name in JOINT_NAMES}
    return {
        name: float(deviation_data.get(str(JOINT_SERVO_CHANNELS[name]), 0) or 0)
        for name in JOINT_NAMES
    }


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


def pulses_for(arm_ik, joints: Dict[str, float]):
    """Ask the SDK what pulses one joint vector corresponds to. Nothing moves."""
    ordered = tuple(joints[name] for name in JOINT_NAMES)
    return arm_ik.transformAngelAdaptArm(*joint_degrees_to_servo_angles(ordered))


def collect_samples(arm_ik) -> Dict[str, List[Tuple[float, float]]]:
    """Sweep each joint on its own and record the pulse the SDK computes."""
    samples: Dict[str, List[Tuple[float, float]]] = {name: [] for name in JOINT_NAMES}
    rejected = 0
    for name in JOINT_NAMES:
        for angle in SWEEP_DEGREES:
            joints = {other: 0.0 for other in JOINT_NAMES}
            joints[name] = float(angle)
            pulses = pulses_for(arm_ik, joints)
            if not pulses:
                rejected += 1
                continue
            pulse = pulses.get(f"servo{JOINT_SERVO_CHANNELS[name]}")
            if pulse is not None:
                samples[name].append((float(angle), float(pulse)))
    if rejected:
        print(f"  ({rejected} swept angles were outside the SDK's servo range)")
    return samples


def check_against_sdk(arm_ik, profile, deviations: Dict[str, float]) -> Tuple[int, float]:
    """Confirm the fitted map reproduces the SDK's own pulses on real poses.

    The fit is derived from single-joint sweeps, so this is the independent
    check that it still holds for the multi-joint vectors real IK produces.
    """
    checked, worst = 0, 0.0
    for x, y, z, pitch in CHECK_POSES:
        solution = hiwonder_ik(x, y, z, pitch)
        if solution is None:
            continue
        expected = arm_ik.transformAngelAdaptArm(*solution)
        if not expected:
            continue
        joints = servo_angles_to_joint_degrees(*solution)
        checked += 1
        for name, angle in zip(JOINT_NAMES, joints):
            channel = JOINT_SERVO_CHANNELS[name]
            ours = profile.servos[name].pulse_for(angle)
            # servosMove writes the IK value plus the trim, so that is what a
            # correct map has to reproduce -- not the raw IK value.
            servo = profile.servos[name]
            # Clamp their value the way we do before comparing: where the SDK
            # would write past the servo's range, the difference is our refusal
            # to do that, not a disagreement about the map.
            theirs = float(expected[f"servo{channel}"]) + deviations[name]
            theirs = max(servo.min_pulse, min(servo.max_pulse, theirs))
            worst = max(worst, abs(ours - theirs))
    return checked, worst


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
    deviations = load_deviations()

    profile = load_profile(args.config)
    servos = dict(profile.servos)
    print(f"{'joint':>10} {'samples':>8} {'centre':>9} {'trim':>7} {'us/deg':>8} {'dir':>5} {'worst err':>10}  status")
    all_verified = True
    for name in JOINT_NAMES:
        points = samples[name]
        try:
            calibration, worst = fit_servo_calibration(
                JOINT_SERVO_CHANNELS[name], points, args.max_residual_us
            )
        except ValueError as exc:
            all_verified = False
            print(f"{name:>10} {len(points):>8} {'-':>9} {'-':>7} {'-':>8} {'-':>5} {'-':>10}  FAILED: {exc}")
            continue
        calibration = replace(calibration, deviation_us=deviations[name])
        servos[name] = calibration
        all_verified = all_verified and calibration.verified
        direction = "+" if calibration.sign > 0 else "-"
        status = "verified" if calibration.verified else f"NOT verified (> {args.max_residual_us} us)"
        print(
            f"{name:>10} {len(points):>8} {calibration.center_pulse:>9.1f} "
            f"{calibration.deviation_us:>+7.0f} {calibration.pulse_per_degree:>8.4f} "
            f"{direction:>5} {worst:>9.2f}us  {status}"
        )

    updated = CalibrationProfile(
        chassis=profile.chassis, arm=profile.arm, gripper=profile.gripper, servos=servos
    )

    print("\nTravel available after trim:")
    for name in JOINT_NAMES:
        low, high = updated.servos[name].reachable_deg()
        lost = (90.0 - high) + (low + 90.0)
        note = f"   ({lost:.1f} deg lost to trim)" if lost > 0.5 else ""
        print(f"  {name:>9}: {low:+.1f} to {high:+.1f} deg{note}")

    checked, worst_pulse = check_against_sdk(arm_ik, updated, deviations)
    print(f"\nCross-check on {checked} real IK poses: worst disagreement with the "
          f"SDK's own pulses is {worst_pulse:.1f} us")
    if worst_pulse > 1.0:
        all_verified = False
        print("  That is too large -- the fitted map does not reproduce the SDK.")

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
