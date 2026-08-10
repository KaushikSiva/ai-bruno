#!/usr/bin/env python3
"""
Interactive arm calibration for the pick-place app.

Jog the gripper in cartesian space, then record the poses the state machine
needs: home, the grasp pose for an object sitting at the camera standoff, and
the drop pose. Recorded values are written back into the `arm_control` block of
config/bruno_config.json.

Run on the robot:

    python3 bruno_apps/pick_place/calibrate_arm.py

Keys:
    w/s   x  +/- (forward / back)
    a/d   y  +/- (left / right)
    r/f   z  +/- (up / down)
    [ ]   step size down / up
    o/c   gripper open / close
    h     go to current home pose
    1     record current pose as HOME
    2     record current pose as GRASP (sets pickup_height)
    3     record current pose as DROP
    p     print current pose
    v     save recorded poses to config/bruno_config.json
    q     quit (arm relaxes to home)
"""

import argparse
import json
import os
import select
import sys
import termios
import tty
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.logging.setup import LOG
from bruno_core.manipulation.arm import ArmConfig, ArmController

CONFIG_PATH = os.path.join(REPO_ROOT, "config", "bruno_config.json")


class KeyReader:
    """Non-blocking single-key reader, same cbreak pattern as the rover app."""

    def __init__(self) -> None:
        self._fd = None
        self._old_attrs = None
        self._enabled = False
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._enabled = True
        except Exception as exc:
            LOG.warning("Terminal key polling unavailable: %s", exc)
            self._enabled = False

    def poll(self, timeout: float = 0.1) -> Optional[str]:
        if not self._enabled or self._fd is None:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None
            ch = sys.stdin.read(1)
            return ch.lower() if ch else None
        except Exception:
            return None

    def close(self) -> None:
        if not self._enabled or self._fd is None or self._old_attrs is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        except Exception:
            pass


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        LOG.warning("Could not read %s: %s", CONFIG_PATH, exc)
        return {}


def save_poses(recorded: dict, gripper_id: Optional[int] = None) -> None:
    """Merge recorded poses into config/bruno_config.json."""
    cfg = load_config()
    arm = cfg.setdefault("arm_control", {})
    if gripper_id is not None:
        arm.setdefault("gripper_servo", {})["id"] = int(gripper_id)
    if "home" in recorded:
        arm["home_position"] = [round(v, 1) for v in recorded["home"]]
    if "drop" in recorded:
        arm["drop_position"] = [round(v, 1) for v in recorded["drop"]]
    if "grasp" in recorded:
        arm["pickup_height"] = round(recorded["grasp"][2], 1)
        arm["grasp_position"] = [round(v, 1) for v in recorded["grasp"]]
    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=4)
    print(f"\r\n💾 saved to {CONFIG_PATH}\r")
    print(f"\r   arm_control: {json.dumps(arm)}\r")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive MasterPi arm calibration")
    parser.add_argument("--step", type=float, default=1.0, help="initial jog step in cm")
    parser.add_argument("--dry-run", action="store_true", help="log motions without moving")
    parser.add_argument(
        "--gripper-id",
        type=int,
        default=None,
        help="override the gripper servo id (config default is often wrong per robot)",
    )
    args = parser.parse_args()

    cfg = ArmConfig.from_dict(load_config())
    if args.gripper_id is not None:
        cfg.gripper_servo_id = args.gripper_id
        print(f"\rgripper servo id override: {cfg.gripper_servo_id}\r")
    arm = ArmController(cfg=cfg, dry_run=args.dry_run)

    x, y, z = cfg.home_position
    step = float(args.step)
    recorded: dict = {}

    print(__doc__)
    print(f"\r\nstarting at home {cfg.home_position}, step {step:.1f} cm\r")
    arm.home()

    keys = KeyReader()
    try:
        while True:
            key = keys.poll(0.1)
            if key is None:
                continue

            nx, ny, nz = x, y, z
            if key == "w":
                nx += step
            elif key == "s":
                nx -= step
            elif key == "a":
                ny += step
            elif key == "d":
                ny -= step
            elif key == "r":
                nz += step
            elif key == "f":
                nz -= step
            elif key == "]":
                step = min(5.0, step + 0.5)
                print(f"\rstep = {step:.1f} cm\r")
                continue
            elif key == "[":
                step = max(0.5, step - 0.5)
                print(f"\rstep = {step:.1f} cm\r")
                continue
            elif key == "o":
                arm.open_gripper()
                print("\rgripper open\r")
                continue
            elif key == "c":
                arm.close_gripper()
                print("\rgripper closed\r")
                continue
            elif key == "h":
                x, y, z = cfg.home_position
                arm.move_to((x, y, z))
                print(f"\rhome ({x:.1f}, {y:.1f}, {z:.1f})\r")
                continue
            elif key == "p":
                print(f"\rpose ({x:.1f}, {y:.1f}, {z:.1f})  step {step:.1f}\r")
                continue
            elif key in ("1", "2", "3"):
                label = {"1": "home", "2": "grasp", "3": "drop"}[key]
                recorded[label] = (x, y, z)
                print(f"\r📌 {label} = ({x:.1f}, {y:.1f}, {z:.1f})\r")
                continue
            elif key == "v":
                if recorded:
                    save_poses(recorded, gripper_id=cfg.gripper_servo_id)
                else:
                    print("\rnothing recorded yet\r")
                continue
            elif key == "q":
                break
            else:
                continue

            if arm.move_to((nx, ny, nz), move_time_ms=500):
                x, y, z = nx, ny, nz
                print(f"\rpose ({x:.1f}, {y:.1f}, {z:.1f})  step {step:.1f}\r")
            else:
                print(f"\r⚠️  ({nx:.1f}, {ny:.1f}, {nz:.1f}) unreachable — staying put\r")
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        arm.relax()
        print("\r\narm relaxed to home\r")
        if recorded:
            print(f"recorded (unsaved unless you pressed v): {recorded}\r")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
