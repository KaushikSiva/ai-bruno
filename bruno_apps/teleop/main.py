#!/usr/bin/env python3
"""
Bruno teleoperation: drive the chassis and arm by hand, in sim or for real.

The same command vocabulary reaches either backend, because both go through
`bruno_core.vla` and share one calibration profile. A run in MuJoCo is meant to
predict the run on hardware, so the two are deliberately kept a single flag
apart:

    # simulation, with the viewer
    python3 bruno_apps/teleop/main.py --target sim keys

    # the real robot, once its pulse map is calibrated
    python3 bruno_apps/teleop/main.py --target real keys

One-shot commands take the same flags:

    python3 bruno_apps/teleop/main.py --target sim rotate_by_deg --angle 45
    python3 bruno_apps/teleop/main.py --target sim rotate_arm_by_deg --joint base --angle 20
    python3 bruno_apps/teleop/main.py --target sim status

Rotation is positive counter-clockwise: +45 turns left, -45 turns right, for
the chassis and for the arm's base joint alike.

Arm state is not persisted between runs: each invocation assumes the arm starts
at the configured home pose, and a joint command writes all four channels. Put
the arm at home first, or lead with `home_arm --duration-ms 2000`.

Pass --robot-url to drive a bridge running elsewhere (the robot on the LAN, or
a simulator in another terminal) instead of opening a driver in this process.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bruno_core.vla.calibration import load_profile
from bruno_core.vla.clients import RobotClient
from bruno_core.vla.contracts import (
    ALLOWED_ACTIONS,
    ARM_ROTATION_ACTION,
    GRIPPER_ACTIONS,
    HOME_ARM_ACTION,
    POSITION_ACTIONS,
    ROTATION_ACTION,
    ZERO_SPEED_ACTIONS,
    Action,
)
from bruno_core.vla.controller import RobotController, RobotNotArmed, limits_from_env
from bruno_core.vla.drivers import make_driver
from bruno_core.vla.kinematics import JOINT_NAMES

LOG = logging.getLogger("bruno_apps.teleop")

TARGET_DRIVERS = {"sim": "mujoco", "real": "bruno", "mock": "mock"}
META_COMMANDS = ("status", "arm", "disarm", "keys")


# --------------------------------------------------------------------------
# Sessions: a local driver and a remote bridge behind one interface.
# --------------------------------------------------------------------------


class LocalSession:
    """Owns a driver and controller in this process."""

    def __init__(self, args: argparse.Namespace):
        profile = load_profile(args.config)
        driver = make_driver(
            TARGET_DRIVERS[args.target],
            profile=profile,
            model_path=args.mujoco_model,
            dry_run=args.dry_run,
            allow_uncalibrated=args.allow_uncalibrated,
        )
        self.driver = driver
        self.profile = profile
        self.controller = RobotController(driver=driver, profile=profile, **limits_from_env())

    def status(self) -> Dict[str, Any]:
        return self.controller.status()

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        return self.controller.set_armed(armed)

    def send(self, action: Action) -> Dict[str, Any]:
        return self.controller.execute(action)

    def close(self) -> None:
        self.controller.close()


class RemoteSession:
    """Talks to a bridge over HTTP."""

    def __init__(self, args: argparse.Namespace):
        self.client = RobotClient(args.robot_url, os.getenv("BRUNO_SHARED_TOKEN", ""))
        self.profile = load_profile(args.config)
        self.driver = None

    def status(self) -> Dict[str, Any]:
        return self.client.status()

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        return self.client.set_armed(armed)

    def send(self, action: Action) -> Dict[str, Any]:
        return self.client.command(action)

    def close(self) -> None:
        pass


class MirrorSession:
    """Drives a primary session and echoes every command to a mirror.

    The point is to watch MuJoCo move while the real robot moves. The robot is
    primary and authoritative: it is commanded first, so a slow or dead mirror
    can never delay it, and every result teleop reports is the robot's. The
    mirror is best effort -- its failures are logged once and swallowed, because
    a simulation that fell over is not a reason to stop being able to stop the
    robot.

    The reverse would be unsafe: if the mirror could veto or delay a command,
    the viewer would be deciding what the hardware does.
    """

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror
        self.profile = primary.profile
        self.driver = getattr(primary, "driver", None)
        self._mirror_failed = False

    def _mirror_call(self, name: str, *call_args) -> None:
        if self.mirror is None:
            return
        try:
            getattr(self.mirror, name)(*call_args)
        except Exception as exc:
            if not self._mirror_failed:
                self._mirror_failed = True
                LOG.warning(
                    "mirror %s failed (%s: %s); the robot keeps running, the viewer "
                    "will now lag behind it", name, type(exc).__name__, exc,
                )
        else:
            self._mirror_failed = False

    def status(self) -> Dict[str, Any]:
        status = dict(self.primary.status())
        if self.mirror is not None:
            try:
                status["mirror"] = self.mirror.status()
            except Exception as exc:
                status["mirror"] = {"error": f"{type(exc).__name__}: {exc}"}
        return status

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        result = self.primary.set_armed(armed)
        self._mirror_call("set_armed", armed)
        return result

    def send(self, action: Action) -> Dict[str, Any]:
        result = self.primary.send(action)
        self._mirror_call("send", action)
        return result

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            if self.mirror is not None:
                try:
                    self.mirror.close()
                except Exception:
                    pass


def open_session(args: argparse.Namespace):
    primary = RemoteSession(args) if args.robot_url else LocalSession(args)
    mirror_url = getattr(args, "mirror_url", "")
    if not mirror_url:
        return primary
    if mirror_url == args.robot_url:
        raise SystemExit("--mirror-url and --robot-url are the same bridge")
    mirror_args = argparse.Namespace(**vars(args))
    mirror_args.robot_url = mirror_url
    return MirrorSession(primary, RemoteSession(mirror_args))


def build_action(args: argparse.Namespace) -> Action:
    payload: Dict[str, Any] = {
        "action": args.command,
        "speed": args.speed,
        "duration_ms": args.duration_ms,
        "angle_deg": args.angle,
        "joint": args.joint or "",
        "confidence": 1.0,
        "reason": "manual teleop command",
    }
    if args.command in ZERO_SPEED_ACTIONS:
        payload["speed"] = 0.0
    return Action.from_dict(payload)


# --------------------------------------------------------------------------
# Interactive keyboard mode
# --------------------------------------------------------------------------

KEY_BINDINGS = {
    "w": ("up", "drive forward"),
    "s": ("down", "drive back"),
    "a": ("left", "strafe left"),
    "d": ("right", "strafe right"),
    "q": ("rotate_ccw", "turn left by the angle step"),
    "e": ("rotate_cw", "turn right by the angle step"),
    "i": ("move_arm_up", "arm up"),
    "k": ("move_arm_down", "arm down"),
    "j": ("move_arm_left", "arm left"),
    "l": ("move_arm_right", "arm right"),
    "u": ("move_arm_forward", "arm out"),
    "o": ("move_arm_backward", "arm in"),
    "1": ("joint:base:-", "base joint clockwise"),
    "2": ("joint:base:+", "base joint counter-clockwise"),
    "3": ("joint:shoulder:-", "shoulder down"),
    "4": ("joint:shoulder:+", "shoulder up"),
    "5": ("joint:elbow:-", "elbow in"),
    "6": ("joint:elbow:+", "elbow out"),
    "7": ("joint:wrist:-", "wrist down"),
    "8": ("joint:wrist:+", "wrist up"),
    "[": ("open_gripper", "open the jaws"),
    "]": ("close_gripper", "close the jaws"),
    "h": (HOME_ARM_ACTION, "return the arm home"),
    " ": ("stop", "stop the chassis"),
}

ARROW_KEYS = {"A": "w", "B": "s", "D": "a", "C": "d"}  # up, down, left, right


def print_help(
    speed: float, duration_ms: int, angle: float, joint_step: float,
    supports_strafe: bool = True,
) -> None:
    print("\n  Bruno teleop")
    print(f"  speed {speed:.2f} | burst {duration_ms} ms | turn {angle:.0f} deg | joint step {joint_step:.0f} deg")
    print("  ------------------------------------------------------------------")
    if supports_strafe:
        print("   w/s/a/d or arrows   drive forward / back / strafe left / right")
    else:
        print("   w/s or up/down      drive forward / back")
        print("   a/d                 strafe -- UNAVAILABLE on this chassis (wheels)")
    print("   q / e               turn left / right by the angle step")
    print("   i/k  j/l  u/o       arm up/down, left/right, out/in")
    print("   1..8                jog base, shoulder, elbow, wrist by degrees")
    print("   [ / ]               open / close the gripper")
    print("   h                   home the arm       space  stop")
    print("   + / -               change speed       < / >  change the angle step")
    print("   ?                   this help          x or Ctrl-C  quit")
    print("  ------------------------------------------------------------------\n")


class RawTerminal:
    """Holds cbreak mode for a whole teleop session.

    Setting and restoring the mode around each individual read looks equivalent
    but is not: keys pressed while output is printing land in a cooked terminal,
    which echoes them and queues them for later. That shows up as stray
    `kkkkkkkk` in the transcript and a burst of commands arriving after the
    fact. cbreak rather than raw, so Ctrl-C stays a signal.
    """

    def __init__(self) -> None:
        self._termios = None
        self._fd = None
        self._saved = None

    def open(self) -> None:
        import termios
        import tty

        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        termios.tcflush(self._fd, termios.TCIFLUSH)  # drop anything already typed

    def close(self) -> None:
        if self._saved is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)
            self._saved = None


def read_key() -> str:
    """Read one keypress, decoding arrows to w/a/s/d.

    Expects the terminal to already be in cbreak mode; see RawTerminal.
    """
    char = sys.stdin.read(1)
    if char == "\x1b":  # escape sequence; arrows arrive as ESC [ A..D
        if sys.stdin.read(1) == "[":
            return ARROW_KEYS.get(sys.stdin.read(1), "")
        return ""
    return char


def action_for_binding(
    binding: str, speed: float, duration_ms: int, angle: float, joint_step: float
) -> Action:
    """Turn one key binding into an Action.

    Module level so the tests can walk every binding in KEY_BINDINGS and check
    it parses. Speed is zeroed only for the actions the contract forbids it on:
    the Cartesian arm jogs need a positive speed, because that is what scales
    their step.
    """
    if binding == "stop":
        return Action.stop("teleop stop")
    if binding.startswith("joint:"):
        _, joint, sign = binding.split(":")
        return Action.from_dict({
            "action": ARM_ROTATION_ACTION, "joint": joint,
            "angle_deg": joint_step if sign == "+" else -joint_step,
            "duration_ms": max(duration_ms, 300), "confidence": 1.0, "reason": "teleop",
        })
    if binding in ("rotate_ccw", "rotate_cw"):
        return Action.from_dict({
            "action": ROTATION_ACTION, "speed": speed,
            "angle_deg": angle if binding == "rotate_ccw" else -angle,
            "duration_ms": duration_ms, "confidence": 1.0, "reason": "teleop",
        })
    return Action.from_dict({
        "action": binding,
        "speed": 0.0 if binding in ZERO_SPEED_ACTIONS else speed,
        "duration_ms": max(duration_ms, 300) if binding in POSITION_ACTIONS else duration_ms,
        "confidence": 1.0, "reason": "teleop",
    })


class Heartbeat(threading.Thread):
    """Keeps the bridge armed while idle, so the dead-man still means something.

    The watchdog exists to stop the robot when teleop dies. Refreshing it with a
    stop only after a quiet period preserves that guarantee while letting a
    human take their time between keypresses.
    """

    def __init__(self, session, idle_seconds: float):
        super().__init__(name="bruno-teleop-heartbeat", daemon=True)
        self.session = session
        self.idle_seconds = idle_seconds
        self.last_command = time.monotonic()
        self._stop = threading.Event()

    def touch(self) -> None:
        self.last_command = time.monotonic()

    def run(self) -> None:
        while not self._stop.wait(0.2):
            if time.monotonic() - self.last_command < self.idle_seconds:
                continue
            try:
                self.session.send(Action.stop("teleop heartbeat"))
                self.session.set_armed(True)
            except Exception:
                pass  # a transient bridge error must not kill the input loop
            self.touch()

    def cancel(self) -> None:
        self._stop.set()


def summarize(status: Dict[str, Any]) -> str:
    driver = status.get("driver", {})
    arm = driver.get("arm", {})
    joints = arm.get("joints_deg", {})
    tcp = arm.get("tcp_estimate_cm", {})
    joint_text = " ".join(f"{name[:2]}{joints.get(name, 0):+.0f}" for name in JOINT_NAMES)
    return (
        f"{'ARMED' if status.get('armed') else 'disarmed'} | "
        f"tcp ({tcp.get('x', 0):+.1f},{tcp.get('y', 0):+.1f},{tcp.get('z', 0):+.1f})cm | "
        f"{joint_text} | grip {driver.get('gripper', '?')}"
    )


def run_keys(session, args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        print("Interactive mode needs a terminal; use a one-shot command instead.", file=sys.stderr)
        return 2

    speed, duration_ms = args.speed, args.duration_ms
    angle, joint_step = args.angle or 15.0, args.joint_step

    supports_strafe = getattr(session.profile.chassis, "supports_strafe", True)
    session.set_armed(True)
    heartbeat = Heartbeat(session, idle_seconds=1.0)
    heartbeat.start()
    print_help(speed, duration_ms, angle, joint_step, supports_strafe)
    terminal = RawTerminal()
    terminal.open()

    try:
        while True:
            key = read_key()
            if key in ("x", "\x03", "\x04"):
                break
            if key == "?":
                print_help(speed, duration_ms, angle, joint_step, supports_strafe)
                continue
            if key in ("+", "="):
                speed = min(1.0, round(speed + 0.05, 2)); print(f"  speed {speed:.2f}"); continue
            if key in ("-", "_"):
                speed = max(0.05, round(speed - 0.05, 2)); print(f"  speed {speed:.2f}"); continue
            if key == ">":
                angle = min(90.0, angle + 5); print(f"  angle step {angle:.0f} deg"); continue
            if key == "<":
                angle = max(5.0, angle - 5); print(f"  angle step {angle:.0f} deg"); continue
            if key not in KEY_BINDINGS:
                continue

            binding, label = KEY_BINDINGS[key]
            try:
                result = session.send(
                    action_for_binding(binding, speed, duration_ms, angle, joint_step)
                )
            except RobotNotArmed:
                print("  disarmed by the watchdog; re-arming")
                session.set_armed(True)
                continue
            except Exception as exc:
                # Driver refusals carry a full explanation; one line is enough
                # here, and the docs hold the rest.
                detail = str(exc).split(",")[0].split(".")[0][:90]
                print(f"  {label}: refused -- {detail}")
                continue
            heartbeat.touch()
            print(f"  {label:<28} {summarize(result)}")
    except KeyboardInterrupt:
        pass
    finally:
        terminal.close()
        heartbeat.cancel()
        try:
            session.send(Action.stop("teleop exit"))
            session.set_armed(False)
        except Exception as exc:
            print(f"  warning: could not disarm cleanly: {exc}", file=sys.stderr)
        print("\n  stopped and disarmed")
    return 0


# --------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate Bruno's chassis and arm in simulation or on hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Rotation is positive counter-clockwise: +45 turns left, -45 turns right.",
    )
    parser.add_argument("command", choices=(*META_COMMANDS, *sorted(ALLOWED_ACTIONS)))
    parser.add_argument("--target", choices=sorted(TARGET_DRIVERS), default="sim",
                        help="sim = MuJoCo, real = the MasterPi, mock = record only")
    parser.add_argument("--robot-url", default=os.getenv("BRUNO_BRIDGE_URL", ""),
                        help="Drive a bridge over HTTP instead of opening a driver here")
    parser.add_argument("--mirror-url", default=os.getenv("MIRROR_URL", ""),
                        help="Echo every command to a second bridge (e.g. a local MuJoCo "
                             "bridge) so simulation mirrors the robot. Best effort: the "
                             "primary target is never delayed or blocked by it")
    parser.add_argument("--speed", type=float, default=0.3, help="Normalized speed, 0-1")
    parser.add_argument("--duration-ms", type=int, default=400, help="Length of one motion burst")
    parser.add_argument("--angle", type=float, default=0.0, help="Signed angle in degrees")
    parser.add_argument("--joint", choices=JOINT_NAMES, help="Joint for rotate_arm_by_deg")
    parser.add_argument("--joint-step", type=float, default=10.0,
                        help="Degrees per joint jog in interactive mode")
    parser.add_argument("--config", default="", help="Path to bruno_config.json")
    parser.add_argument("--mujoco-model", default="", help="Override the MasterPi MJCF path")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --target real, log motions instead of executing them")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Permit joint-space arm commands with an unverified pulse map")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import logging

    args = parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"),
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command == ARM_ROTATION_ACTION and not args.joint:
        print("rotate_arm_by_deg needs --joint", file=sys.stderr)
        return 2

    session = open_session(args)
    try:
        if args.command == "keys":
            return run_keys(session, args)
        if args.command == "status":
            print(json.dumps(session.status(), indent=2, sort_keys=True))
            return 0
        if args.command in ("arm", "disarm"):
            print(json.dumps(session.set_armed(args.command == "arm"), indent=2, sort_keys=True))
            return 0

        action = build_action(args)
        if action.action != "stop":
            session.set_armed(True)
        result = session.send(action)
        # A one-shot velocity burst has to outlive the process that issued it.
        if action.action not in POSITION_ACTIONS and action.action != "stop":
            executed = result.get("executed", {})
            time.sleep(executed.get("duration_ms", action.duration_ms) / 1000.0 + 0.2)
        print(json.dumps(session.status(), indent=2, sort_keys=True))
        return 0
    except RobotNotArmed as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            session.send(Action.stop("teleop finished"))
            session.set_armed(False)
        except Exception:
            pass
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
