from dataclasses import dataclass
from typing import Final

from bruno_core.config.env import load_env, get_env_int, get_env_float


@dataclass(frozen=True)
class Settings:
    camera_retry_attempts: int
    camera_retry_delay: float

    ultra_caution_cm: float
    ultra_danger_cm: float
    forward_speed: int
    turn_speed: int
    turn_time: float
    backup_time: float

    photo_interval: int
    summary_delay: int


def load_settings() -> Settings:
    """Load .env and return immutable runtime settings."""
    load_env()
    photo_interval: Final[int] = get_env_int('PHOTO_INTERVAL_SEC', 15)
    summary_delay: Final[int] = get_env_int('SUMMARY_DELAY_SEC', 120)
    return Settings(
        camera_retry_attempts=3,
        camera_retry_delay=2.0,
        ultra_caution_cm=get_env_float('ULTRA_CAUTION_CM', 50.0),
        ultra_danger_cm=get_env_float('ULTRA_DANGER_CM', 25.0),
        forward_speed=get_env_int('BRUNO_SPEED', 40),
        turn_speed=get_env_int('BRUNO_TURN_SPEED', 40),
        turn_time=get_env_float('BRUNO_TURN_TIME', 0.5),
        backup_time=get_env_float('BRUNO_BACKUP_TIME', 0.0),
        photo_interval=photo_interval,
        summary_delay=summary_delay,
    )
