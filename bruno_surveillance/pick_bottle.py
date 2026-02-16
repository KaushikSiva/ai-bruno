#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Bottle/Can Pickup (External Camera + Maverick)

Standalone script that uses the external USB camera, Groq's Meta Llama‑4
Maverick vision model for object detection, and the MasterPi SDK to pick up
a bottle or can IF it is already close by. The robot does not drive or rotate.

Requirements
- MasterPi SDK available at /home/pi/MasterPi
- External USB camera accessible via OpenCV (v4l2)
- GROQ_API_KEY set for vision detection

Usage
  python3 -m bruno_surveillance.pick_bottle

Environment overrides (optional)
- CAM_MODE=external (default is external)
- PICK_CAUTION_CM=35  PICK_DANGER_CM=20  PICK_TARGET="bottle or can"
- ARM_GRIPPER_ID=5 ARM_GRIPPER_OPEN=1800 ARM_GRIPPER_CLOSED=1200
 - CLOSE_SIZE_RATIO=0.18  PICK_Y=8  PICK_Z=8
"""

import os
import sys
import time
from typing import Optional, Tuple

sys.path.append('/home/pi/MasterPi')

import cv2  # type: ignore

from utils import LOG, save_image_path
from camera_shared import make_camera, read_or_reconnect
from ultrasonic import UltrasonicRGB

# Hardware imports
try:
    import common.mecanum as mecanum  # type: ignore
    from common.ros_robot_controller_sdk import Board  # type: ignore
    from kinematics.arm_move_ik import ArmIK  # type: ignore
    HW_OK = True
except Exception as e:  # pragma: no cover
    LOG.warning(f"Hardware SDK not available: {e}")
    HW_OK = False
    mecanum = None
    Board = None
    ArmIK = None

# Groq Maverick detector
from groq_vision import detect_object_bbox


class BottlePicker:
    def __init__(self):
        self.mode = os.environ.get('CAM_MODE', 'external')  # force external behavior
        if self.mode != 'external':
            LOG.info("Overriding camera mode to 'external' as requested.")
            self.mode = 'external'

        # Detection target
        self.target = os.environ.get('PICK_TARGET', 'bottle or can')

        # Safety thresholds (in cm)
        self.caution_cm = float(os.environ.get('PICK_CAUTION_CM', '35'))
        self.danger_cm = float(os.environ.get('PICK_DANGER_CM', '20'))
        self.pick_cm = float(os.environ.get('PICK_PICKUP_CM', '25'))
        self.close_size_ratio = float(os.environ.get('CLOSE_SIZE_RATIO', '0.18'))

        # Movement params
        self.forward_speed = int(os.environ.get('PICK_FWD_SPEED', '35'))
        self.turn_speed = int(os.environ.get('PICK_TURN_SPEED', '35'))

        # Gripper params
        self.grip_id = int(os.environ.get('ARM_GRIPPER_ID', '5'))
        self.grip_open = int(os.environ.get('ARM_GRIPPER_OPEN', '1800'))
        self.grip_closed = int(os.environ.get('ARM_GRIPPER_CLOSED', '1200'))

        # Initialize hardware
        self.car = mecanum.MecanumChassis() if HW_OK else None
        self.board = Board() if HW_OK else None
        self.AK = ArmIK() if HW_OK else None
        if self.AK and self.board:
            self.AK.board = self.board
            # camera_external may have disabled PWM; re-enable for arm/gripper
            try:
                for ch in (1, 2, 3, 4, 5, 6):
                    self.board.pwm_servo_enable(ch, True)
            except Exception:
                pass

        # Ultrasonic safety
        self.ultra = UltrasonicRGB()

        # Camera
        self.camera = make_camera(self.mode, retry_attempts=3, retry_delay=2.0)

        # Runtime state
        self.frame_w = 0
        self.frame_h = 0

    # ---------- Hardware helpers ----------
    def stop(self):
        if not self.car:
            return
        try:
            self.car.set_velocity(0, 0, 0)
        except Exception:
            pass

    def rotate(self, right: bool, duration: float = 0.3):
        if not self.car:
            time.sleep(duration)
            return
        try:
            yaw = self.turn_speed if right else -self.turn_speed
            # yaw-only rotation
            self.car.set_velocity(0, 0, yaw)
            time.sleep(duration)
        except Exception:
            pass
        finally:
            self.stop()

    def forward_burst(self, duration: float = 0.5):
        if not self.car:
            time.sleep(duration)
            return
        try:
            # 90 degrees is forward for mecanum
            self.car.set_velocity(self.forward_speed, 90, 0)
            time.sleep(duration)
        except Exception:
            pass
        finally:
            self.stop()

    def open_gripper(self, t: float = 0.4):
        if self.board:
            try:
                self.board.pwm_servo_set_position(t, [[self.grip_id, self.grip_open]])
                time.sleep(t)
            except Exception:
                pass

    def close_gripper(self, t: float = 0.6):
        if self.board:
            try:
                self.board.pwm_servo_set_position(t, [[self.grip_id, self.grip_closed]])
                time.sleep(t)
            except Exception:
                pass

    def arm_home(self):
        if self.AK:
            try:
                # Safe neutral pose per HiWonder examples
                self.AK.setPitchRangeMoving((0, 6, 18), 0, -90, 90, 1500)
                time.sleep(1.6)
            except Exception:
                pass

    def arm_pre_grasp(self):
        if self.AK:
            try:
                # Slightly lower and forward from home
                self.AK.setPitchRangeMoving((0, 7, 12), -30, -90, 0, 900)
                time.sleep(1.0)
            except Exception:
                pass

    def arm_grasp_height(self):
        if self.AK:
            try:
                # Lower closer to object (tweak as needed for your setup)
                z = max(5, int(os.environ.get('PICK_Z', '8')))
                self.AK.setPitchRangeMoving((0, 8, z), -50, -90, 0, 900)
                time.sleep(1.0)
            except Exception:
                pass

    # ---------- Vision and control ----------
    def _center_offset(self, bbox) -> float:
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        cx = x + (w / 2.0)
        if self.frame_w <= 0:
            return 0.0
        return (cx / float(self.frame_w)) - 0.5  # left negative, right positive

    def _size_ratio(self, bbox) -> float:
        if self.frame_w <= 0:
            return 0.0
        return bbox['w'] / float(self.frame_w)

    def _save_frame(self, frame) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        path = save_image_path('pick_bottle')
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return str(path)

    def _read_distance(self) -> Optional[float]:
        try:
            return self.ultra.get_distance_cm()
        except Exception:
            return None

    def _ready_to_pick(self, bbox) -> Tuple[bool, Optional[float]]:
        """Decide if the target is close enough to pick without moving the base."""
        dist = self._read_distance()
        if dist is not None:
            LOG.info(f"Ultrasonic distance: {dist:.1f} cm")
            return (dist <= self.pick_cm), dist
        # Fallback on apparent size if ultrasonic not available
        sr = self._size_ratio(bbox)
        LOG.info(f"No distance reading; using size ratio {sr:.3f}")
        return (sr >= self.close_size_ratio), None

    def _x_offset_cm(self, bbox) -> int:
        """Map horizontal pixel offset to a small lateral X offset in cm for ArmIK."""
        off = self._center_offset(bbox)  # -0.5..0.5 typical
        # Map ±0.5 -> ±6 cm (clamped)
        x_cm = int(max(-6.0, min(6.0, off * 12.0)))
        return x_cm

    def _do_pick(self, bbox=None):
        LOG.info("Starting pick sequence")
        self.stop()
        # Ensure servos are enabled
        if self.board:
            try:
                for ch in (1, 2, 3, 4, 5, 6):
                    self.board.pwm_servo_enable(ch, True)
            except Exception:
                pass
        self.open_gripper(0.4)
        self.arm_home()
        # Choose lateral offset from bbox if available
        x_cm = 0
        try:
            if bbox is not None:
                x_cm = self._x_offset_cm(bbox)
        except Exception:
            x_cm = 0
        # Pre‑grasp over target
        if self.AK:
            try:
                y_pre = int(os.environ.get('PICK_Y', '8'))
                z_pre = max(10, int(os.environ.get('PICK_Z', '8')) + 4)
                self.AK.setPitchRangeMoving((x_cm, y_pre, z_pre), -30, -90, 0, 900)
                time.sleep(1.0)
            except Exception:
                pass
        # Lower to grasp
        self.arm_grasp_height()
        self.close_gripper(0.6)
        # Lift back up
        self.arm_home()
        LOG.info("Pick sequence complete")

    def run(self):
        LOG.info("🤖 Bruno Bottle/Can Pickup (External Camera + Maverick)")
        if not self.camera.open():
            LOG.error("Failed to open external camera")
            return

        last_detect_t = 0.0
        DETECT_INTERVAL = 1.0
        last_good_frame = None
        try:
            while True:
                # Read or reconnect
                try:
                    ok, frame = self.camera.read()
                except Exception:
                    ok, frame = False, None
                if not ok or frame is None:
                    frame = read_or_reconnect(self.camera, last_good_frame)
                    if frame is None:
                        time.sleep(0.1)
                        continue
                last_good_frame = frame

                self.frame_h, self.frame_w = frame.shape[:2]

                now = time.time()
                if now - last_detect_t < DETECT_INTERVAL:
                    # Throttle detections to reduce API usage
                    time.sleep(0.02)
                    continue
                last_detect_t = now

                img_path = self._save_frame(frame)
                bbox = detect_object_bbox(img_path, target=self.target)
                if not bbox:
                    LOG.info(f"No {self.target} detected; standing by")
                    continue

                ready, dist = self._ready_to_pick(bbox)
                if ready:
                    self._do_pick(bbox)
                    break
                else:
                    LOG.info("Target not close enough; waiting without moving")
        except KeyboardInterrupt:
            LOG.info("Interrupted by user")
        finally:
            self.stop()
            try:
                self.camera.release()
            except Exception:
                pass


def main():
    bp = BottlePicker()
    bp.run()


if __name__ == '__main__':
    main()
