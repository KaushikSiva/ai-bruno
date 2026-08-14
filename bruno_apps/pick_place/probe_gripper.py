#!/usr/bin/env python3
"""
Identify and tune the gripper servo empirically.

MasterPi drives everything over PWM: functions/color_sorting.py opens the jaws
with [[1, 2000]], closes with [[1, 1500]], and resets the arm with
[[3, 515], [4, 2170], [5, 945]]. So id 1 is the gripper and 3-6 are arm joints,
which is what config/bruno_config.json ships with.

Use this when the jaws don't respond, or to tune the open/closed pulses for the
object you're picking:

    python3 bruno_apps/pick_place/probe_gripper.py                # walk the ids
    python3 bruno_apps/pick_place/probe_gripper.py --only 1       # just the jaws

Answering "yes" to the jaws moving drops into a jog loop for the travel limits,
which are written back to config/bruno_config.json.

Ids 3-6 are skipped by default because they move the arm, not the jaws; pass
--include-arm-ids to sweep them anyway, with the arm supported.

A caveat worth knowing before you trust a negative result: these writes are
fire-and-forget. A servo that is unplugged, wired backwards, or dead accepts
every command and raises nothing. Silence here is not proof the id is wrong —
it once cost this project a long hunt for a software bug that turned out to be
a reversed connector.
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

# These move arm joints, not the jaws (color_sorting.py's arm reset).
ARM_SERVO_IDS = (3, 4, 5, 6)


def candidates(include_arm_ids: bool) -> list:
    return [i for i in range(1, 7) if include_arm_ids or i not in ARM_SERVO_IDS]


def drive(board, servo_id: int, pulse: int, duration: float) -> None:
    board.pwm_servo_set_position(duration, [[servo_id, int(pulse)]])


def wiggle(board, servo_id: int, center: int, delta: int, duration: float) -> bool:
    """Sweep a channel around center. Returns False if the channel rejected it."""
    try:
        for pulse in (center, center + delta, center - delta, center):
            drive(board, servo_id, pulse, duration)
            time.sleep(duration + 0.25)
        return True
    except Exception as exc:
        print(f"    id {servo_id}: command failed ({exc})")
        return False


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def save(servo_id: int, open_pulse: int, closed_pulse: int) -> None:
    try:
        with open(CONFIG_PATH, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        LOG.warning("Could not read %s: %s", CONFIG_PATH, exc)
        cfg = {}
    gripper = cfg.setdefault("arm_control", {}).setdefault("gripper_servo", {})
    gripper["id"] = int(servo_id)
    gripper["open_position"] = int(open_pulse)
    gripper["closed_position"] = int(closed_pulse)
    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=4)
    print(f"\n💾 saved gripper_servo: {json.dumps(gripper)} -> {CONFIG_PATH}")


def find_limits(board, servo_id: int, duration: float) -> tuple:
    """Jog the gripper to the open and closed pulses you want."""
    pulse = 1500
    step = 100
    print(
        "\nNow set the travel limits.\n"
        "  +/-  jog by the current step      [ ]  step down / up\n"
        "  o    record this as OPEN          c    record this as CLOSED\n"
        "  d    done\n"
        "Record CLOSED while gripping the object you actually pick, not on empty\n"
        "air, and stay off the hard stops — a stalled servo will cook itself."
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
            drive(board, servo_id, pulse, duration)
            time.sleep(duration + 0.1)
        except Exception as exc:
            print(f"    command failed: {exc}")
    return open_pulse, closed_pulse


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify and tune the MasterPi gripper servo")
    parser.add_argument("--center", type=int, default=1500, help="neutral pulse to sweep around")
    parser.add_argument("--delta", type=int, default=250, help="sweep amplitude in pulse units")
    parser.add_argument("--duration", type=float, default=0.4, help="move time per step, seconds")
    parser.add_argument("--only", type=int, default=None, help="probe a single servo id")
    parser.add_argument(
        "--include-arm-ids",
        action="store_true",
        help="also probe ids 3-6 (these drive the arm joints — support the arm first)",
    )
    args = parser.parse_args()

    ids = [args.only] if args.only is not None else candidates(args.include_arm_ids)

    board = Board()
    try:
        board.enable_reception()
    except Exception as exc:
        LOG.debug("enable_reception unavailable: %s", exc)

    print(__doc__)
    print(f"Sweeping {args.center} ± {args.delta} on each id. 'q' aborts.\n")

    for servo_id in ids:
        key = ask(f"  probe id {servo_id}? [enter=go / s=skip / q=quit] ")
        if key == "q":
            print("aborted")
            return 1
        if key == "s":
            continue
        if not wiggle(board, servo_id, args.center, args.delta, args.duration):
            continue
        if ask("    did the JAWS move? [y/N] ").startswith("y"):
            print(f"\n✅ gripper is pwm servo id {servo_id}")
            open_pulse, closed_pulse = find_limits(board, servo_id, args.duration)
            if open_pulse is None or closed_pulse is None:
                print("no open/closed pair recorded — nothing saved")
                return 1
            save(servo_id, open_pulse, closed_pulse)
            return 0

    print("\nNo id moved the jaws — suspect wiring before software.")
    print("Check the jaw servo's lead at the PWM 1 header: seated, and not reversed.")
    print("To isolate a dead header from a dead servo, swap the jaw servo's lead with")
    print("the arm joint on PWM 3 and re-run; whichever moves tells you which is at fault.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
