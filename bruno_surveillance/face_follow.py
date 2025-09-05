#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Face Follow (standalone)

Behavior
- Opens camera (builtin/external via --mode)
- Detects a face via MediaPipe FaceDetection and tracks its center
- Greets once (first confirmed face) with nod + optional "Hello"
- Follows by centering on the face and maintaining distance
- Uses ultrasonic to avoid getting too close

Run
  python3 bruno_surveillance/face_follow.py --mode external --audio --voice Ashley

Dependencies
- mediapipe, opencv-python
- MasterPi SDK for motion & ultrasonic
- bruno_surveillance/audio_tts.py for TTS (optional)
"""
import os
import sys
import time
import argparse

sys.path.append('/home/pi/MasterPi')

import cv2
import mediapipe as mp

from utils import LOG, paths
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
        # Greeting + tracking state
        self.greeted_once = False
        self.last_bbox = None  # (x, y, w, h) for last detected face

        # Face detector (MediaPipe)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.75
        )

        # Follow tuning
        # If your chassis turns the opposite way, set invert_yaw = False/True
        self.invert_yaw = True  # many chassis map yaw inverted; flip if needed
        self.center_dead_px = 30  # face needs tighter centering than legs
        # Use relative area thresholds (ratio of face box to frame area)
        # These values work across common resolutions (e.g., 640x480)
        self.area_ratio_far = 0.010   # <1.0% of frame: move forward
        self.area_ratio_target = 0.030  # ~3% of frame: good distance target
        self.area_ratio_near = 0.060  # >6% of frame: too close, stop

        # Timing
        self.loop_sleep = 0.03

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
        if self.greeted_once:
            return
        self.greeted_once = True
        try:
            self._nod()
            if self.speaker:
                self.speaker.speak_sync("Hello")
        except Exception:
            pass

    def _follow_step(self, frame_w, frame_h, bbox):
        # Safety: ultrasonic
        d = self.ultra.get_distance_cm()
        if d is not None and d <= 25:
            self.motion.stop(); return

        (x, y, w, h) = bbox
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        frame_area = max(1, frame_w * frame_h)
        area_ratio = area / frame_area

        # Heading control (discrete)
        dx = cx - (frame_w // 2)
        if dx < -self.center_dead_px:
            # target is to the left of center
            if self.invert_yaw:
                self.motion.turn_right(0.12)
            else:
                self.motion.turn_left(0.12)
        elif dx > self.center_dead_px:
            # target is to the right of center
            if self.invert_yaw:
                self.motion.turn_left(0.12)
            else:
                self.motion.turn_right(0.12)

        # Range control (based on relative face size)
        if area_ratio < self.area_ratio_far:
            self.motion.forward(0.2)
        elif area_ratio > self.area_ratio_near:
            self.motion.stop()
        else:
            # in band: gentle forward if still below target
            if area_ratio < self.area_ratio_target:
                self.motion.forward(0.1)
            else:
                self.motion.stop()

    def _detect_face_bbox(self, frame_bgr) -> tuple | None:
        """Detect the most confident face and return pixel bbox (x,y,w,h) or None."""
        try:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.face_detection.process(rgb)
            if not res.detections:
                return None
            best = None
            best_area = 0
            for det in res.detections:
                score = det.score[0] if det.score else 0.0
                if score < 0.7:
                    continue
                rb = det.location_data.relative_bounding_box
                x = max(0, int(rb.xmin * w))
                y = max(0, int(rb.ymin * h))
                ww = max(0, int(rb.width * w))
                hh = max(0, int(rb.height * h))
                if ww <= 0 or hh <= 0:
                    continue
                area = ww * hh
                if area > best_area:
                    best = (x, y, ww, hh)
                    best_area = area
            return best
        except Exception:
            return None

    def run(self):
        LOG.info('🙂 Face Follow (MediaPipe): greet + nod + follow')
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
                # Detect face every loop; keep last bbox for smoothing when transiently lost
                box = self._detect_face_bbox(last_frame)
                if box is not None and box[2] > 0 and box[3] > 0:
                    self.last_bbox = box
                    if not self.greeted_once:
                        self._greet()
                if self.last_bbox is not None:
                    self._follow_step(w, h, self.last_bbox)
                else:
                    self.motion.stop()

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
