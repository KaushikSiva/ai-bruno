#!/usr/bin/env python3
"""
Find the gripper servo empirically.

The arm's IK moves fine but the jaws don't, which means the configured gripper
channel is wrong — either the id or the protocol (PWM vs serial bus). This
walks the candidate channels one at a time and asks you which one moved the
jaws, then writes the answer into config/bruno_config.json.

Run on the robot, with the arm clear of obstacles:

    python3 bruno_apps/pick_place/probe_gripper.py

Arm joints are driven by ArmIK over the bus at ids 3-6, so those are skipped by
default — probing them would jog the arm, not the jaws. Use --include-arm-ids
only if you have the arm supported and want to rule them out.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.append("/home/pi/MasterPi")

from bruno_core.logging.setup import LOG  # noqa: E402

try:
    from common.ros_robot_controller_sdk import Board  # type: ignore
except Exception as exc:  # pragma: no cover - hardware-only path
    print(f"MasterPi SDK not importable ({exc}) — run this on the robot.")
    raise SystemExit(1)

CONFIG_PATH = os.path.join(REPO_ROOT, "config", "bruno_config.json")

# ArmIK owns these over the bus; touching them moves the arm, not the jaws.
ARM_BUS_IDS = (3, 4, 5, 6)


def candidates(include_arm_ids: bool) -> list:
    ids = list(range(1, 7))
    pwm = [("pwm", i) for i in ids if include_arm_ids or i not in ARM_BUS_IDS]
    bus = [("bus", i) for i in ids if include_arm_ids or i not in ARM_BUS_IDS]
    return pwm + bus


def drive(board, protocol: str, servo_id: int, pulse: int, duration: float) -> None:
    """Send one position command, letting failures surface."""
    if protocol == "pwm":
        board.pwm_servo_set_position(duration, [[servo_id, int(pulse)]])
    else:
        board.bus_servo_set_position(duration, [[servo_id, int(pulse)]])


def wiggle(board, protocol: str, servo_id: int, center: int, delta: int, duration: float) -> bool:
    """Sweep a channel around center. Returns False if the channel rejected it."""
    try:
        for pulse in (center, center + delta, center - delta, center):
            drive(board, protocol, servo_id, pulse, duration)
            time.sleep(duration + 0.25)
        return True
    except Exception as exc:
        print(f"    {protocol} id {servo_id}: command failed ({exc})")
        return False


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def save(protocol: str, servo_id: int, open_pulse: int, closed_pulse: int) -> None:
    try:
        with open(CONFIG_PATH, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        LOG.warning("Could not read %s: %s", CONFIG_PATH, exc)
        cfg = {}
    gripper = cfg.setdefault("arm_control", {}).setdefault("gripper_servo", {})
    gripper["id"] = int(servo_id)
    gripper["protocol"] = protocol
    gripper["open_position"] = int(open_pulse)
    gripper["closed_position"] = int(closed_pulse)
    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=4)
    print(f"\n💾 saved gripper_servo: {json.dumps(gripper)} -> {CONFIG_PATH}")


def find_limits(board, protocol: str, servo_id: int, duration: float) -> tuple:
    """Jog the found channel to the open and closed pulses you like."""
    pulse = 1500
    step = 100
    print(
        "\nNow set the travel limits.\n"
        "  +/-  jog by the current step      [ ]  step down / up\n"
        "  o    record this as OPEN          c    record this as CLOSED\n"
        "  d    done\n"
        "Keep the jaws off their hard stops — a stalled servo will cook itself."
    )
    open_pulse = closed_pulse = None
    while True:
        key = ask(f"  pulse={pulse} step={step} open={open_pulse} closed={closed_pulse} > ")
        if key in ("+", "="):
            pulse = min(2500, pulse + step)
        elif key == "-":
            pulse = max(500, pulse - step)
        elif key == "]":
            step = min(200, step + 25)
            continue
        elif key == "[":
            step = max(25, step - 25)
            continue
        elif key == "o":
            open_pulse = pulse
            continue
        elif key == "c":
            closed_pulse = pulse
            continue
        elif key == "d":
            break
        else:
            continue
        try:
            drive(board, protocol, servo_id, pulse, duration)
            time.sleep(duration + 0.1)
        except Exception as exc:
            print(f"    command failed: {exc}")
    return open_pulse, closed_pulse


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify the MasterPi gripper servo channel")
    parser.add_argument("--center", type=int, default=1500, help="neutral pulse to sweep around")
    parser.add_argument("--delta", type=int, default=250, help="sweep amplitude in pulse units")
    parser.add_argument("--duration", type=float, default=0.4, help="move time per step, seconds")
    parser.add_argument(
        "--include-arm-ids",
        action="store_true",
        help="also probe bus ids 3-6 (these drive the arm joints — support the arm first)",
    )
    args = parser.parse_args()

    board = Board()
    try:
        board.enable_reception()
    except Exception as exc:
        LOG.debug("enable_reception unavailable: %s", exc)

    print(__doc__)
    print(f"Sweeping {args.center} ± {args.delta} on each channel. 'q' aborts.\n")

    for protocol, servo_id in candidates(args.include_arm_ids):
        key = ask(f"  probe {protocol} id {servo_id}? [enter=go / s=skip / q=quit] ")
        if key == "q":
            print("aborted")
            return 1
        if key == "s":
            continue
        if not wiggle(board, protocol, servo_id, args.center, args.delta, args.duration):
            continue
        if ask("    did the JAWS move? [y/N] ").startswith("y"):
            print(f"\n✅ gripper is {protocol} servo id {servo_id}")
            open_pulse, closed_pulse = find_limits(board, protocol, servo_id, args.duration)
            if open_pulse is None or closed_pulse is None:
                print("no open/closed pair recorded — nothing saved")
                return 1
            save(protocol, servo_id, open_pulse, closed_pulse)
            return 0

    print("\nNo channel moved the jaws.")
    print("Check the jaw servo's cable at the controller, and re-run with --include-arm-ids")
    print("(arm supported) to rule out the bus ids ArmIK uses.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
