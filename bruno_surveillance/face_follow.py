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
        # Groq Vision tracking state
        self.greeted_once = False
        self.next_groq_check = 0.0
        self.next_track_at = 0.0
        self.track_interval = 0.8  # seconds between Groq leg detections
        self.last_bbox = None

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

    def _groq_track_legs(self, frame_bgr) -> tuple | None:
        """Call Groq to detect lower-body bbox and return (x,y,w,h) or None."""
        try:
            # Save a smaller frame to speed up IO
            small = cv2.resize(frame_bgr, (480, int(frame_bgr.shape[0] * 480 / max(1, frame_bgr.shape[1]))))
            probe = paths.debug / 'ff_track.jpg'
            cv2.imwrite(str(probe), small)
            try:
                from bruno_surveillance.groq_vision import detect_person_legs
            except Exception:
                from groq_vision import detect_person_legs  # type: ignore
            bbox = detect_person_legs(str(probe))
            if not bbox:
                return None
            # bbox is in pixel space of the saved image; we can approximate same size as small
            return (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
        except Exception:
            return None

    def run(self):
        LOG.info('🦵 Leg Follow (Groq Vision): greet + nod + follow')
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
                # Greeter: lightweight caption probe
                rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

                # Greet once using Groq vision (every few seconds)
                now = time.time()
                if (not self.greeted_once) and (now >= self.next_groq_check):
                    self.next_groq_check = now + self.groq_interval
                    if self._groq_person_probe(last_frame):
                        self._greet()

                # Tracking via Groq at a lower rate; hold last bbox in between
                now = time.time()
                if now >= self.next_track_at:
                    self.next_track_at = now + self.track_interval
                    box = self._groq_track_legs(last_frame)
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
