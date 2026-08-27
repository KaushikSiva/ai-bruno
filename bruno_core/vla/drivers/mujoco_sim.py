"""MuJoCo-backed MasterPi driver.

Runs the pinned ``third_party/masterpi-mujoco`` model behind the same
``BaseDriver`` dispatch the hardware uses, so a command produces the same joint
targets in both and only the delivery differs. MuJoCo is imported lazily: the
robot itself never needs it installed.

Frames. The MJCF has ``+x`` forward and ``+y`` to the robot's left, while the
Hiwonder arm frame this repo commands in has ``+x`` to the right and ``+y``
forward. The swap lives in :meth:`_drive` and nowhere else.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..calibration import CalibrationProfile
from ..kinematics import JOINT_ACTUATORS, JOINT_NAMES, JointDegrees
from .base import BaseDriver

LOG = logging.getLogger("bruno_core.vla.drivers.mujoco")

DRIVE_ACTUATORS = ("drive_vx", "drive_vy", "drive_wz")
GRIPPER_ACTUATOR = "servo1_gripper"
MODEL_JOINTS = ("joint6_base_yaw", "joint5_shoulder", "joint4_elbow", "joint3_wrist")
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
DEFAULT_CAMERA = "wrist_cam"


def default_model_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "third_party" / "masterpi-mujoco" / "masterpi.xml"


class MuJoCoDriver(BaseDriver):
    name = "mujoco"

    def __init__(
        self,
        profile: Optional[CalibrationProfile] = None,
        model_path: str = "",
        realtime: bool = True,
    ):
        super().__init__(profile)
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is not installed; run: pip install -r requirements-sim.txt"
            ) from exc

        chosen = Path(model_path or os.getenv("BRUNO_MUJOCO_MODEL", "") or default_model_path())
        self.model_path = chosen.expanduser().resolve()
        if not self.model_path.is_file():
            raise RuntimeError(
                f"MasterPi MJCF not found at {self.model_path}; run: "
                "git submodule update --init --recursive"
            )

        self._mujoco = mujoco
        try:
            self._model = mujoco.MjModel.from_xml_path(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(f"could not load {self.model_path}: {exc}") from exc
        self._data = mujoco.MjData(self._model)

        actuators = {self._model.actuator(i).name: i for i in range(self._model.nu)}
        required = (*DRIVE_ACTUATORS, *JOINT_ACTUATORS.values(), GRIPPER_ACTUATOR)
        missing = [name for name in required if name not in actuators]
        if missing:
            raise RuntimeError(f"MJCF is missing actuators: {', '.join(missing)}")
        self._drive_ids = tuple(actuators[name] for name in DRIVE_ACTUATORS)
        self._arm_ids = tuple(actuators[JOINT_ACTUATORS[name]] for name in JOINT_NAMES)
        self._gripper_id = actuators[GRIPPER_ACTUATOR]
        self._joint_adr = {
            name: int(self._model.joint(name).qposadr[0])
            for name in (*MODEL_JOINTS, *BASE_JOINTS)
        }

        self._realtime = realtime
        self._closed = threading.Event()
        self._set_joints(self.arm.joint_tuple(), self.profile.arm.move_time_ms)
        self._set_gripper(True, 0)
        mujoco.mj_forward(self._model, self._data)

        self._physics = threading.Thread(
            target=self._step_loop, name="bruno-mujoco-physics", daemon=True
        )
        self._physics.start()
        LOG.info("MuJoCo simulation driver ready: %s", self.model_path)

    # ---------- BaseDriver hooks ----------

    def _drive(self, vx_cmps: float, vy_cmps: float, wz_deg_s: float) -> None:
        """Drive at a body-frame velocity, as the real chassis does.

        The MJCF gives the chassis ``base_x``/``base_y`` slide joints *before*
        ``base_yaw``, so their axes stay world-fixed: writing the actuators
        directly would send the robot along a fixed world axis no matter which
        way it is pointing. Rotating the command by the current yaw restores
        the body-frame behaviour of a real mecanum base, without touching the
        pinned model.

        Simultaneous translation and rotation would still drift as the yaw
        changes underneath the command, but the action contract issues one or
        the other, never both.
        """
        # Command frame is +x right / +y forward; the model is +x forward / +y left.
        forward, left = vy_cmps / 100.0, -vx_cmps / 100.0
        yaw = float(self._data.qpos[self._joint_adr["base_yaw"]])
        world_x = forward * math.cos(yaw) - left * math.sin(yaw)
        world_y = forward * math.sin(yaw) + left * math.cos(yaw)
        self._set_ctrl(self._drive_ids, (world_x, world_y, math.radians(wz_deg_s)))

    def _set_joints(self, joints_deg: JointDegrees, duration_ms: int) -> None:
        # Position actuators servo toward the target; duration_ms is advisory
        # here because the MJCF's kp/kv set the approach rate.
        self._set_ctrl(self._arm_ids, (math.radians(angle) for angle in joints_deg))

    def _set_gripper(self, opened: bool, duration_ms: int) -> None:
        self._data.ctrl[self._gripper_id] = self.profile.gripper.opening_m_for(opened)

    def _extra_status(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "sim_time_s": round(float(self._data.time), 3),
            "running": not self._closed.is_set(),
            "base_pose": {
                "x_m": round(float(self._data.qpos[self._joint_adr["base_x"]]), 4),
                "y_m": round(float(self._data.qpos[self._joint_adr["base_y"]]), 4),
                "yaw_deg": round(
                    math.degrees(float(self._data.qpos[self._joint_adr["base_yaw"]])), 2
                ),
            },
            "measured_joints_deg": self.measured_joints_deg(),
            "telemetry": self.telemetry(),
        }

    # ---------- simulation-only extras ----------

    def measured_joints_deg(self) -> Dict[str, float]:
        """True joint angles from the model, to compare against the command.

        Hardware has no feedback path, so this is the one place a tracking error
        between commanded and achieved joints can actually be observed.
        """
        return {
            name: round(math.degrees(float(self._data.qpos[self._joint_adr[model_joint]])), 2)
            for name, model_joint in zip(JOINT_NAMES, MODEL_JOINTS)
        }

    def joint_tracking_error_deg(self) -> Dict[str, float]:
        measured = self.measured_joints_deg()
        return {
            name: round(measured[name] - self.arm.joints[name], 2) for name in JOINT_NAMES
        }

    def telemetry(self) -> Dict[str, Any]:
        left = self._rangefinder_cm("sonar_left")
        right = self._rangefinder_cm("sonar_right")
        distances = [value for value in (left, right) if value is not None]
        result: Dict[str, Any] = {
            "sonar_left_cm": left,
            "sonar_right_cm": right,
            "sim_time_s": round(float(self._data.time), 3),
        }
        if distances:
            result["distance_cm"] = round(min(distances), 2)
        return result

    def render_camera_jpeg(
        self, camera_name: str = DEFAULT_CAMERA, width: int = 640, height: int = 480, quality: int = 85
    ) -> bytes:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for simulated frames") from exc
        width = max(64, min(1920, int(width)))
        height = max(64, min(1080, int(height)))
        renderer = None
        with self._lock:
            try:
                renderer = self._mujoco.Renderer(self._model, height=height, width=width)
                renderer.update_scene(self._data, camera=camera_name)
                pixels = renderer.render()
            except Exception as exc:
                raise RuntimeError(f"could not render camera {camera_name!r}: {exc}") from exc
            finally:
                if renderer is not None:
                    renderer.close()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=max(20, min(95, quality)))
        return buffer.getvalue()

    def run_viewer(self) -> None:
        """Run the passive viewer on the calling thread (macOS needs mjpython)."""
        try:
            from mujoco import viewer
        except ImportError as exc:
            raise RuntimeError("the installed MuJoCo package has no viewer") from exc
        # launch_passive builds its first scene from _data, and the physics
        # thread has been stepping since __init__. Reading mjData while mj_step
        # writes it segfaults, intermittently, before the window ever appears --
        # so hold the same lock the step loop takes across construction, not
        # just around sync(). _lock is an RLock, so sync() below may retake it.
        with self._lock:
            passive_cm = viewer.launch_passive(self._model, self._data)
            passive = passive_cm.__enter__()
        try:
            while passive.is_running() and not self._closed.wait(1.0 / 60.0):
                with self._lock:
                    passive.sync()
        finally:
            passive_cm.__exit__(None, None, None)

    def close(self) -> None:
        self._closed.set()
        self._physics.join(timeout=2.0)

    # ---------- internals ----------

    def _step_loop(self) -> None:
        timestep = float(self._model.opt.timestep)
        next_step = time.monotonic()
        while not self._closed.is_set():
            with self._lock:
                self._mujoco.mj_step(self._model, self._data)
            if not self._realtime:
                continue
            next_step += timestep
            delay = next_step - time.monotonic()
            if delay > 0:
                self._closed.wait(delay)
            else:
                next_step = time.monotonic()

    def _set_ctrl(self, ids: Iterable[int], values: Iterable[float]) -> None:
        for actuator_id, value in zip(ids, values):
            self._data.ctrl[actuator_id] = value

    def _rangefinder_cm(self, name: str) -> Optional[float]:
        try:
            sensor = self._model.sensor(name)
        except Exception:
            return None
        value_m = float(self._data.sensordata[int(sensor.adr[0])])
        return round(value_m * 100.0, 2) if value_m >= 0.0 else None
