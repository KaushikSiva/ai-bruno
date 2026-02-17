import time
from dataclasses import dataclass
from typing import Optional

from bruno_core.logging.setup import LOG


@dataclass
class SafetyState:
    emergency_start: Optional[float] = None
    emergency_reversed_once: bool = False


class SafetyController:
    """
    Encapsulates ultrasonic safety behavior: stop in danger, avoid in caution,
    and drive forward when safe. Manages emergency timers and a one-time reverse.
    """

    def __init__(self, ultra_caution_cm: float, ultra_danger_cm: float, turn_time: float, backup_time: float):
        self.ultra_caution_cm = ultra_caution_cm
        self.ultra_danger_cm = ultra_danger_cm
        self.turn_time = turn_time
        self.backup_time = backup_time
        self.state = SafetyState()

    def handle(self, distance_cm: Optional[float], frame_idx: int, motion, ultra) -> None:
        d = distance_cm
        if d is not None and d <= self.ultra_danger_cm:
            if self.state.emergency_start is None:
                self.state.emergency_start = time.time()
                self.state.emergency_reversed_once = False
                LOG.warning(f'EMERGENCY STOP ({d:.1f} cm) — timer started')
            else:
                LOG.warning(f'EMERGENCY STOP ({d:.1f} cm) — ongoing')
            ultra.set_rgb(255, 0, 0)
            motion.stop()
            time.sleep(0.02)
            elapsed = time.time() - self.state.emergency_start
            if (elapsed >= 5.0) and (not self.state.emergency_reversed_once):
                LOG.warning('⏪ EMERGENCY >5s — reversing one step')
                motion.reverse_burst(duration=0.6)
                self.state.emergency_reversed_once = True
            if elapsed >= 30.0:
                raise SystemExit('prolonged emergency stop (>30s)')
            return

        if d is not None and d <= self.ultra_caution_cm:
            ultra.set_rgb(255, 180, 0)
            if self.state.emergency_start is not None:
                LOG.info('✅ Left EMERGENCY zone — timer reset')
            self.state.emergency_start = None
            self.state.emergency_reversed_once = False
            if self.backup_time > 0:
                motion.forward(duration=self.backup_time)
                motion.stop()
            left = ((frame_idx // 60) % 2 == 0)
            if left: motion.turn_left(self.turn_time)
            else:    motion.turn_right(self.turn_time)
            return

        # Safe
        ultra.set_rgb(0, 255, 0)
        if self.state.emergency_start is not None:
            LOG.info('✅ Safe distance — emergency cleared')
        self.state.emergency_start = None
        self.state.emergency_reversed_once = False
        motion.forward()
