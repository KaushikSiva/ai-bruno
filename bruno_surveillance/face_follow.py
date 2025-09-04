#!/usr/bin/env python3
# coding: utf-8
"""
Bruno Leg Follow (standalone)

Behavior
- Opens camera (builtin/external via --mode)
- Detects a person via MediaPipe Pose and tracks lower body (legs/hips)
- Greets once (Groq Vision caption or first pose) with nod + "Hello"
- Follows from behind by centering on legs and maintaining distance
- Uses ultrasonic to avoid getting too close

Run
  python3 bruno_surveillance/face_follow.py --mode external --audio --voice Ashley

Dependencies
- mediapipe, opencv-python
- MasterPi SDK for motion & ultrasonic
- bruno_surveillance/groq_vision.py for caption probe
- bruno_surveillance/audio_tts.py for TTS (Inworld)
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
        # Pose for leg/hip landmarks
        self.pose = mp.solutions.pose.Pose(model_complexity=0,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.greeted_once = False
        self.next_groq_check = 0.0

        # Follow tuning
        # If your chassis turns the opposite way, set invert_yaw = False/True
        self.invert_yaw = True  # many chassis map yaw inverted; flip if needed
        self.center_dead_px = 50
        # Area proxy based on lower-body bounding box; tune for your FOV
        self.area_near = 130000   # stop if larger than this (too close)
        self.area_target = 90000  # try to reach this area
        self.area_far = 50000     # move forward if below this

        # Timing
        self.loop_sleep = 0.03
        self.groq_interval = 2.0

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

        # Heading control (discrete)
        dx = cx - (frame_w // 2)
        if dx < -self.center_dead_px:
            # face is to the left of center
            if self.invert_yaw:
                self.motion.turn_right(0.12)
            else:
                self.motion.turn_left(0.12)
        elif dx > self.center_dead_px:
            # face is to the right of center
            if self.invert_yaw:
                self.motion.turn_left(0.12)
            else:
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

    def _groq_person_probe(self, bgr_frame) -> bool:
        """Use groq_vision to decide if a person is present in this frame."""
        try:
            try:
                # Try absolute import first (script run as module)
                from bruno_surveillance import groq_vision as gv
            except Exception:
                from groq_vision import get_caption as _gc  # type: ignore
                gv = None
            # Downscale to speed up save/transfer
            small = cv2.resize(bgr_frame, (480, int(bgr_frame.shape[0] * 480 / max(1, bgr_frame.shape[1]))))
            probe = paths.debug / 'ff_groq_probe.jpg'
            cv2.imwrite(str(probe), small)
            caption = gv.get_caption(str(probe)) if gv else _gc(str(probe))
            caps = caption.lower()
            keywords = ("person", "people", "man", "woman", "boy", "girl", "human")
            return any(k in caps for k in keywords)
        except Exception:
            return False

    def _lower_body_bbox(self, h, w, lmk) -> tuple | None:
        """Compute a bbox around hips/knees/ankles from pose landmarks.
        Returns (x, y, bw, bh) in pixels, or None if invalid.
        """
        idx = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles
        xs, ys = [], []
        for i in idx:
            lm = lmk.landmark[i]
            if lm.visibility < 0.4:
                continue
            xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
        if len(xs) < 2 or len(ys) < 2:
            return None
        x1, x2 = max(0, min(xs)), min(w-1, max(xs))
        y1, y2 = max(0, min(ys)), min(h-1, max(ys))
        # Ensure bbox has reasonable size (avoid zero-area)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        # Focus on lower third: pad upward slightly to include thighs
        y1 = max(0, y1 - int(0.05 * h))
        return (x1, y1, bw, bh)

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
                res = self.pose.process(rgb)

                # Greet once using Groq vision (every few seconds)
                now = time.time()
                if (not self.greeted_once) and (now >= self.next_groq_check):
                    self.next_groq_check = now + self.groq_interval
                    if self._groq_person_probe(last_frame):
                        self._greet()

                if res.pose_landmarks:
                    box = self._lower_body_bbox(h, w, res.pose_landmarks)
                    if box is not None:
                        if not self.greeted_once:
                            self._greet()
                        self._follow_step(w, h, box)
                    else:
                        self.motion.stop()
                else:
                    self.motion.stop()
                    # Do not reset greeted_once; greet only one time per session

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
