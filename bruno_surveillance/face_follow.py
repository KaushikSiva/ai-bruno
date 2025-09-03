#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Face Follow (standalone)

Behavior
- Opens camera (builtin/external via --mode)
- Detects faces (MediaPipe)
- On first detection after a gap: nod head and say "Hello" (voice)
- Follows the person by keeping the face centered and at a target size
- Uses ultrasonic to avoid getting too close

Run
  python3 bruno_surveillance/face_follow.py --mode external --audio --voice Ashley

Dependencies
- mediapipe, opencv-python
- MasterPi SDK for motion & ultrasonic
- bruno_surveillance/audio_tts.py for TTS (Inworld)
"""
import os
import sys
import time
import argparse

sys.path.append('/home/pi/MasterPi')

import cv2
import mediapipe as mp

from utils import LOG
from camera_shared import make_camera, read_or_reconnect
from robot_motion import MecanumWrapper
from ultrasonic import UltrasonicRGB
from audio_tts import TTSSpeaker

try:
    from common.ros_robot_controller_sdk import Board
    HW_BOARD = True
except Exception:
    Board = None
    HW_BOARD = False


class FaceFollower:
    def __init__(self, mode: str, audio: bool, voice: str):
        self.camera = make_camera(mode, retry_attempts=3, retry_delay=2.0)
        self.motion = MecanumWrapper(forward_speed=40, turn_speed=40)
        self.ultra = UltrasonicRGB()
        self.speaker = TTSSpeaker(enabled=audio, voice=voice) if audio else None
        if self.speaker:
            self.speaker.start()

        self.board = Board() if HW_BOARD else None
        self.face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8)
        self.last_seen = 0.0
        self.greeted = False

        # Follow tuning
        self.center_dead_px = 40
        self.area_near = 110000   # stop if larger than this (too close)
        self.area_target = 80000  # try to reach this area
        self.area_far = 40000     # move forward if below this

        # Timing
        self.loop_sleep = 0.03
        self.greet_cooldown = 6.0

    def _nod(self):
        if not self.board:
            return
        try:
            # simple nod on servo 3
            for v, t in ((500, 0.15), (900, 0.15), (500, 0.15), (700, 0.15)):
                self.board.pwm_servo_set_position(0.1, [[3, v]])
                time.sleep(t)
        except Exception:
            pass

    def _greet(self):
        now = time.time()
        if self.greeted and (now - self.last_seen) < self.greet_cooldown:
            return
        self.greeted = True
        try:
            self._nod()
            if self.speaker:
                self.speaker.speak_sync("Hello")
        except Exception:
            pass

    def _follow_step(self, frame_w, frame_h, face_box):
        # Safety: ultrasonic
        d = self.ultra.get_distance_cm()
        if d is not None and d <= 25:
            self.motion.stop(); return

        (x, y, w, h) = face_box
        cx = x + w // 2
        cy = y + h // 2
        area = w * h

        # Heading control (discrete)
        dx = cx - (frame_w // 2)
        if dx < -self.center_dead_px:
            self.motion.turn_left(0.12)
        elif dx > self.center_dead_px:
            self.motion.turn_right(0.12)

        # Range control
        if area < self.area_far:
            self.motion.forward(0.2)
        elif area > self.area_near:
            self.motion.stop()
        else:
            # in band: gentle forward if still below target
            if area < self.area_target:
                self.motion.forward(0.1)
            else:
                self.motion.stop()

    def run(self):
        LOG.info('🤝 Face Follow: greet + nod + follow')
        if not self.camera.open():
            LOG.error('❌ Cannot open camera')
            return
        last_frame = None
        try:
            while True:
                last_frame = read_or_reconnect(self.camera, last_frame)
                if last_frame is None:
                    time.sleep(self.loop_sleep); continue
                h, w = last_frame.shape[:2]
                rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                res = self.face.process(rgb)

                if res.detections:
                    # take the best face
                    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
                    score = det.score[0] if det.score else 0.0
                    if score >= 0.7:
                        bbox = det.location_data.relative_bounding_box
                        box = (
                            int(bbox.xmin * w),
                            int(bbox.ymin * h),
                            int(bbox.width * w),
                            int(bbox.height * h),
                        )
                        self._greet()
                        self.last_seen = time.time()
                        self._follow_step(w, h, box)
                    else:
                        self.motion.stop()
                else:
                    self.motion.stop()
                    self.greeted = False  # next time will greet again

                time.sleep(self.loop_sleep)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.motion.stop()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description='Bruno Face Follow (standalone)')
    p.add_argument('--mode', choices=['builtin','external'], default=os.environ.get('CAM_MODE','external'))
    p.add_argument('--audio', action='store_true', help='Enable greeting voice')
    p.add_argument('--voice', default=os.environ.get('BRUNO_AUDIO_VOICE','Ashley'))
    args = p.parse_args()

    app = FaceFollower(args.mode, args.audio, args.voice)
    app.run()


if __name__ == '__main__':
    main()

