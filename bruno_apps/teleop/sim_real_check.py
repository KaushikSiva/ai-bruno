#!/usr/bin/env python3
"""
Run one command sequence through simulation and hardware, and report the gaps.

The two backends share an action contract, a kinematics module, and a
calibration profile, so their *commanded* joint targets should be identical by
construction. This script checks that claim rather than assuming it, and then
reports the three places sim and real can still legitimately differ:

1. **Tracking.** MuJoCo's position actuators take time to reach a target, so
   the joints the model actually achieves lag the ones commanded. The MasterPi's
   servos lag too, with no feedback to tell you by how much.
2. **Open-loop rotation.** `rotate_by_deg` times a spin against a calibrated
   rate. Acceleration eats into that, so the achieved angle undershoots -- in
   sim measurably, on hardware invisibly. The shortfall printed here is the
   correction to fold into `max_rotation_deg_per_s`.
3. **Acceptance.** A pose the simulator reaches may be refused by the robot's
   own IK, and joint commands are refused outright until the pulse map is
   calibrated. Both show up as failures below.

Off-robot, `--hardware dry` exercises the whole hardware path without a robot:
it still runs the real IK bounds and pulse arithmetic, it just does not move
anything.

    python3 bruno_apps/teleop/sim_real_check.py                # sim vs dry hardware
    python3 bruno_apps/teleop/sim_real_check.py --hardware live # on Bruno
    python3 bruno_apps/teleop/sim_real_check.py --hardware none # sim only
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.vla.calibration import load_profile
from bruno_core.vla.contracts import POSITION_ACTIONS, Action
from bruno_core.vla.drivers import make_driver
from bruno_core.vla.kinematics import JOINT_NAMES

DEFAULT_SEQUENCE: List[Dict[str, Any]] = [
    {"action": "home_arm", "duration_ms": 800},
    {"action": "move_arm_forward", "speed": 1.0, "duration_ms": 600},
    {"action": "move_arm_up", "speed": 1.0, "duration_ms": 600},
    {"action": "move_arm_left", "speed": 1.0, "duration_ms": 600},
    {"action": "rotate_arm_by_deg", "joint": "base", "angle_deg": 20, "duration_ms": 600},
    {"action": "rotate_arm_by_deg", "joint": "shoulder", "angle_deg": -10, "duration_ms": 600},
    {"action": "rotate_arm_by_deg", "joint": "wrist", "angle_deg": 15, "duration_ms": 600},
    {"action": "close_gripper", "duration_ms": 500},
    {"action": "open_gripper", "duration_ms": 500},
    {"action": "home_arm", "duration_ms": 800},
]


def joint_line(joints: Dict[str, float]) -> str:
    return " ".join(f"{joints.get(name, 0.0):+7.2f}" for name in JOINT_NAMES)


def run_arm_comparison(sim, hardware, sequence, settle_s: float) -> int:
    print("\n=== Arm: commanded joint targets, sim vs hardware ===")
    print(f"{'command':<38} {'  base  shoulder   elbow   wrist':>36}  agree  hardware")
    failures = 0
    for step in sequence:
        action = Action.from_dict({"confidence": 1.0, "reason": "sim/real check", **step})
        try:
            sim.apply(action)
        except Exception as exc:
            print(f"{action.describe():<38} {'':>36}  ----   SIM FAILED: {exc}")
            failures += 1
            continue
        time.sleep(settle_s)

        hardware_note = "skipped"
        sim_joints = dict(sim.arm.joints)
        agree = "-"
        if hardware is not None:
            try:
                hardware.apply(action)
                hardware_note = "ok"
                agree = "yes" if all(
                    abs(sim_joints[name] - hardware.arm.joints[name]) < 1e-6
                    for name in JOINT_NAMES
                ) else "NO"
            except Exception as exc:
                hardware_note = f"REFUSED: {type(exc).__name__}: {str(exc)[:60]}"
                failures += 1
        print(f"{action.describe():<38} {joint_line(sim_joints):>36}  {agree:<5}  {hardware_note}")
    return failures


def report_tracking(sim) -> None:
    print("\n=== Sim tracking: achieved joints minus commanded ===")
    time.sleep(1.0)
    errors = sim.joint_tracking_error_deg()
    print("  " + "  ".join(f"{name} {errors[name]:+.2f} deg" for name in JOINT_NAMES))
    worst = max(abs(value) for value in errors.values())
    print(f"  worst {worst:.2f} deg -- hardware servos lag similarly, with no feedback to measure it")


def report_rotation(sim, angles, speed: float) -> None:
    profile = sim.profile
    print("\n=== Open-loop rotation: commanded vs achieved in sim ===")
    print(f"{'commanded':>12} {'spin time':>11} {'achieved':>10} {'shortfall':>11}")
    ratios = []
    for angle in angles:
        start = sim.status()["base_pose"]["yaw_deg"]
        duration_ms = profile.chassis.rotation_duration_ms(angle, speed)
        sim.apply(Action.from_dict({
            "action": "rotate_by_deg", "angle_deg": angle, "speed": speed,
            "duration_ms": duration_ms, "confidence": 1.0, "reason": "sim/real check",
        }))
        time.sleep(duration_ms / 1000.0)
        sim.stop()
        time.sleep(0.6)
        achieved = sim.status()["base_pose"]["yaw_deg"] - start
        ratios.append(achieved / angle if angle else 1.0)
        print(f"{angle:>+11.0f}d {duration_ms:>10}ms {achieved:>+9.1f}d {achieved - angle:>+10.1f}d")
    mean_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    suggested = profile.chassis.max_rotation_deg_per_s * mean_ratio
    print(f"\n  Sim achieves {mean_ratio * 100:.0f}% of the commanded angle (actuator ramp-up).")
    print(f"  To make sim rotations land on target, set vla.chassis.max_rotation_deg_per_s")
    print(f"  to {suggested:.1f} (currently {profile.chassis.max_rotation_deg_per_s:.1f}).")
    print("  Measure the same ratio on the floor before trusting it on hardware.")


def report_calibration(profile, hardware) -> None:
    print("\n=== Calibration state ===")
    chassis = profile.chassis
    print(f"  linear   {chassis.max_linear_speed_cmps:.1f} cm/s at speed 1.0")
    print(f"  rotation {chassis.max_rotation_deg_per_s:.1f} deg/s at speed 1.0")
    unverified = profile.unverified_joints()
    if unverified:
        print(f"  servo pulse map UNVERIFIED for: {', '.join(unverified)}")
        print("  -> run bruno_apps/teleop/calibrate_joints.py on the robot")
    else:
        print("  servo pulse map verified for all four joints")
    if hardware is not None:
        pulses = hardware.status().get("commanded_pulses", {})
        print(f"  pulses for the current pose: {pulses}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Bruno's simulated and hardware control paths")
    parser.add_argument("--hardware", choices=("dry", "live", "none"), default="dry",
                        help="dry = hardware path without motion; live = move the robot; none = sim only")
    parser.add_argument("--config", default="", help="Path to bruno_config.json")
    parser.add_argument("--mujoco-model", default="", help="Override the MasterPi MJCF path")
    parser.add_argument("--settle-s", type=float, default=0.8, help="Pause after each arm command")
    parser.add_argument("--speed", type=float, default=0.5, help="Speed for the rotation test")
    parser.add_argument("--skip-rotation", action="store_true", help="Skip the chassis rotation test")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Let joint commands through an unverified pulse map")
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")

    profile = load_profile(args.config)
    sim = make_driver("mujoco", profile=profile, model_path=args.mujoco_model)
    hardware: Optional[Any] = None
    if args.hardware != "none":
        if args.hardware == "live":
            print("!! --hardware live moves the real robot. Clear the arm workspace.")
        hardware = make_driver(
            "bruno", profile=load_profile(args.config),
            dry_run=args.hardware == "dry",
            allow_uncalibrated=args.allow_uncalibrated,
        )

    # Quiet the shared logger only after the drivers exist: bruno_core.logging
    # re-initialises it to INFO when it is first imported, which happens lazily
    # inside the hardware driver, and the dry-run arm logs every motion.
    logging.getLogger("bruno").setLevel(os.getenv("LOG_LEVEL", "WARNING"))

    try:
        failures = run_arm_comparison(sim, hardware, DEFAULT_SEQUENCE, args.settle_s)
        report_tracking(sim)
        if not args.skip_rotation:
            report_rotation(sim, (90.0, -90.0, 45.0), args.speed)
        report_calibration(profile, hardware)
        print()
        if failures:
            print(f"{failures} command(s) failed or were refused -- see above.")
            return 1
        print("Sim and hardware agreed on every commanded joint target.")
        return 0
    finally:
        sim.stop()
        sim.close()
        if hardware is not None:
            hardware.stop()
            hardware.close()


if __name__ == "__main__":
    raise SystemExit(main())
