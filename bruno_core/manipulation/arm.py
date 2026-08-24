"""
Arm + gripper control for the MasterPi 4-DOF manipulator.

Wraps `kinematics.arm_move_ik.ArmIK` for cartesian moves and the Board PWM
servo API for the gripper. All poses are (x, y, z) in cm in the MasterPi arm
frame; defaults come from the `arm_control` block of config/bruno_config.json.

Hardware is optional: when the MasterPi SDK is unavailable the controller runs
in dry mode, logging every motion instead of executing it. That keeps the
pick-place state machine testable off-robot.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

sys.path.append("/home/pi/MasterPi")

from bruno_core.logging.setup import LOG

try:
    from kinematics.arm_move_ik import ArmIK  # type: ignore
    from common.ros_robot_controller_sdk import Board  # type: ignore

    HW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - hardware-only path
    LOG.warning(f"Arm SDK not available, running dry: {exc}")
    ArmIK = None  # type: ignore
    Board = None  # type: ignore
    HW_AVAILABLE = False


Pose = Tuple[float, float, float]


@dataclass
class ArmConfig:
    """Geometry and servo limits for the manipulator."""

    home_position: Pose = (15.0, 0.0, 20.0)
    drop_position: Pose = (20.0, -15.0, 10.0)
    # Filled in by calibrate_arm.py; falls back to (home x, 0, pickup_height).
    grasp_position: Optional[Pose] = None
    pickup_height: float = 5.0
    approach_clearance: float = 8.0
    carry_height: float = 18.0
    # "overhead" descends onto the object — right for a block, but for a bottle
    # taller than approach_clearance the waypoint above the grasp point sits
    # inside the bottle. "side" holds the grasp height and closes in from
    # approach_backoff cm short of it instead.
    grasp_approach: str = "side"
    approach_backoff: float = 6.0
    # PWM servo 1 is the gripper, 3-6 are the arm joints — see MasterPi's
    # functions/color_sorting.py, which opens the jaws with [[1, 2000]], closes
    # with [[1, 1500]], and resets the arm with [[3, 515], [4, 2170], [5, 945]].
    gripper_servo_id: int = 1
    gripper_open_pulse: int = 2000
    gripper_closed_pulse: int = 1500
    gripper_move_time: float = 0.6
    grasp_pitch: int = -90
    pitch_range: Tuple[int, int] = (-90, 90)
    move_time_ms: int = 1200
    settle_s: float = 0.25

    @classmethod
    def from_dict(cls, cfg: dict) -> "ArmConfig":
        """Build from the `arm_control` block of bruno_config.json."""
        arm = (cfg or {}).get("arm_control", {}) or {}
        gripper = arm.get("gripper_servo", {}) or {}
        defaults = cls()
        home = arm.get("home_position") or list(defaults.home_position)
        drop = arm.get("drop_position") or list(defaults.drop_position)
        grasp = arm.get("grasp_position")
        return cls(
            home_position=(float(home[0]), float(home[1]), float(home[2])),
            drop_position=(float(drop[0]), float(drop[1]), float(drop[2])),
            grasp_position=(float(grasp[0]), float(grasp[1]), float(grasp[2])) if grasp else None,
            pickup_height=float(arm.get("pickup_height", defaults.pickup_height)),
            grasp_approach=str(arm.get("grasp_approach", defaults.grasp_approach)).lower(),
            approach_backoff=float(arm.get("approach_backoff", defaults.approach_backoff)),
            grasp_pitch=int(arm.get("grasp_pitch", defaults.grasp_pitch)),
            gripper_servo_id=int(gripper.get("id", defaults.gripper_servo_id)),
            gripper_open_pulse=int(gripper.get("open_position", defaults.gripper_open_pulse)),
            gripper_closed_pulse=int(gripper.get("closed_position", defaults.gripper_closed_pulse)),
        )


@dataclass
class ArmController:
    """Cartesian moves + gripper for the MasterPi arm."""

    cfg: ArmConfig = field(default_factory=ArmConfig)
    board: Optional[object] = None
    arm_ik: Optional[object] = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.enabled = HW_AVAILABLE and not self.dry_run
        self.last_pose: Optional[Pose] = None
        self.gripper_closed = False
        if not self.enabled:
            LOG.info("🦾 ArmController in dry mode (no motion will be executed)")
            return
        try:
            if self.board is None:
                self.board = Board()
                # MasterPi's function scripts start the serial receive thread
                # before issuing servo commands; without it writes are dropped.
                try:
                    self.board.enable_reception()
                except Exception as exc:
                    LOG.debug(f"enable_reception unavailable: {exc}")
            if self.arm_ik is None:
                self.arm_ik = ArmIK()
            self.arm_ik.board = self.board
            LOG.info(
                f"🦾 ArmController ready (Board + ArmIK), gripper = pwm id "
                f"{self.cfg.gripper_servo_id} "
                f"[{self.cfg.gripper_closed_pulse}..{self.cfg.gripper_open_pulse}]"
            )
        except Exception as exc:
            LOG.error(f"Arm init failed, falling back to dry mode: {exc}")
            self.enabled = False

    # ---------- primitives ----------

    def move_to(self, pose: Pose, move_time_ms: Optional[int] = None, pitch: Optional[int] = None) -> bool:
        """Move the gripper to a cartesian pose. Returns False if unreachable."""
        x, y, z = (float(pose[0]), float(pose[1]), float(pose[2]))
        ms = int(move_time_ms if move_time_ms is not None else self.cfg.move_time_ms)
        p = int(pitch if pitch is not None else self.cfg.grasp_pitch)
        if not self.enabled:
            LOG.info(f"[dry] move_to ({x:.1f}, {y:.1f}, {z:.1f}) pitch={p} t={ms}ms")
            self.last_pose = (x, y, z)
            return True
        try:
            result = self.arm_ik.setPitchRangeMoving(
                (x, y, z), p, self.cfg.pitch_range[0], self.cfg.pitch_range[1], ms
            )
        except Exception as exc:
            LOG.error(f"IK move failed for ({x:.1f}, {y:.1f}, {z:.1f}): {exc}")
            return False
        if not result:
            LOG.warning(f"Pose ({x:.1f}, {y:.1f}, {z:.1f}) unreachable at pitch {p}")
            return False
        # setPitchRangeMoving returns (servos, alpha, movetime) and does not block.
        actual_ms = ms
        try:
            if isinstance(result, Sequence) and len(result) >= 3 and result[2]:
                actual_ms = int(result[2])
        except Exception:
            pass
        time.sleep(actual_ms / 1000.0 + self.cfg.settle_s)
        self.last_pose = (x, y, z)
        return True

    def set_servos(self, targets: Sequence[Sequence[int]], move_time: Optional[float] = None) -> bool:
        """Drive one or more PWM channels to raw pulse widths together.

        `targets` is a sequence of `(channel, pulse)` pairs. Each channel gets
        its own `pwm_servo_set_position` call sharing one move time, which is
        exactly what MasterPi's `ArmIK.servosMove` does -- it issues four
        separate writes rather than one batched list. Matching the SDK here
        removes a whole class of "moves through ArmIK but not through us"
        puzzles. The single sleep at the end still lets the channels travel
        together.
        """
        t = float(move_time if move_time is not None else self.cfg.gripper_move_time)
        pairs = [[int(channel), int(pulse)] for channel, pulse in targets]
        if not self.enabled:
            LOG.info(f"[dry] pwm servos -> {pairs} over {t:.2f}s")
            return True
        try:
            for pair in pairs:
                self.board.pwm_servo_set_position(t, [pair])
            time.sleep(t + 0.1)
            return True
        except Exception as exc:
            # Note the write itself is fire-and-forget: a servo that is unwired,
            # or wired backwards, raises nothing here and silently does nothing.
            LOG.error(f"Servo move failed ({pairs}): {exc}")
            return False

    def set_gripper(self, pulse: int, move_time: Optional[float] = None) -> bool:
        """Drive the gripper servo to a raw pulse width. Returns False on failure."""
        return self.set_servos([(self.cfg.gripper_servo_id, int(pulse))], move_time)

    def open_gripper(self) -> bool:
        ok = self.set_gripper(self.cfg.gripper_open_pulse)
        self.gripper_closed = False
        return ok

    def close_gripper(self) -> bool:
        ok = self.set_gripper(self.cfg.gripper_closed_pulse)
        self.gripper_closed = True
        return ok

    def home(self) -> bool:
        """Return to the home pose with the gripper open."""
        ok = self.move_to(self.cfg.home_position)
        self.open_gripper()
        return ok

    def relax(self) -> None:
        """Park at home and release the gripper — safe end state."""
        self.move_to(self.cfg.home_position, move_time_ms=1500)
        self.open_gripper()

    # ---------- composites ----------

    def pick(self, pose: Pose) -> bool:
        """Approach, close on the object, and lift to carry height.

        The approach waypoint is above the grasp point for "overhead", or short
        of it at the same height for "side" — the latter is what a bottle needs,
        since descending onto one means driving through it.

        Returns False if any waypoint is unreachable; the arm is returned to a
        safe height on failure so the base can re-approach.
        """
        x, y, z = (float(pose[0]), float(pose[1]), float(pose[2]))
        if self.cfg.grasp_approach == "side":
            approach = (max(1.0, x - self.cfg.approach_backoff), y, z)
        else:
            approach = (x, y, z + self.cfg.approach_clearance)
        LOG.info(f"🦾 pick at ({x:.1f}, {y:.1f}, {z:.1f}) via {self.cfg.grasp_approach} approach")
        self.open_gripper()
        if not self.move_to(approach):
            return False
        if not self.move_to((x, y, z), move_time_ms=900):
            self.move_to(approach)
            return False
        self.close_gripper()
        if not self.move_to((x, y, self.cfg.carry_height), move_time_ms=1000):
            return False
        return True

    def place(self, pose: Optional[Pose] = None) -> bool:
        """Lower at the drop pose, release, and retreat upward."""
        x, y, z = pose if pose is not None else self.cfg.drop_position
        x, y, z = float(x), float(y), float(z)
        LOG.info(f"🦾 place at ({x:.1f}, {y:.1f}, {z:.1f})")
        above = (x, y, z + self.cfg.approach_clearance)
        if not self.move_to(above):
            return False
        if not self.move_to((x, y, z), move_time_ms=900):
            return False
        self.open_gripper()
        self.move_to(above)
        return True
