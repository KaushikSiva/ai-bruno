"""Motion drivers: mock, MuJoCo simulation, and the physical MasterPi."""

from .base import BaseDriver, MockDriver, MotionDriver

__all__ = ["BaseDriver", "MockDriver", "MotionDriver", "make_driver"]


def make_driver(name: str, **kwargs):
    """Build a driver by name, importing backend dependencies only on demand."""
    if name == "mock":
        return MockDriver(kwargs.get("profile"))
    if name == "mujoco":
        from .mujoco_sim import MuJoCoDriver

        return MuJoCoDriver(
            profile=kwargs.get("profile"),
            model_path=kwargs.get("model_path", ""),
            realtime=kwargs.get("realtime", True),
        )
    if name == "bruno":
        from .hardware import HardwareDriver

        return HardwareDriver(
            profile=kwargs.get("profile"),
            dry_run=kwargs.get("dry_run", False),
            allow_uncalibrated=kwargs.get("allow_uncalibrated", False),
        )
    raise ValueError(f"unknown driver: {name}")
